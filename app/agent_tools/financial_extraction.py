"""
LangGraph tools for financial data extraction and XBRL analysis.
"""
import os
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import logging

from app.services.report import IngestorService
from app.agent_tools.xbrl_analyzer import XBRLAnalyzer, XBRLAnalysisResult
from app.db.companies import resolve_bse_scripcode

logger = logging.getLogger("financial_tools")


class ShareholdingPatternInput(BaseModel):
    """Input schema for shareholding pattern extraction"""
    company_query: str = Field(..., description="Company name, ticker symbol, or BSE code to analyze")
    from_year: int = Field(..., description="Start year for analysis (e.g., 2022)")
    to_year: int = Field(..., description="End year for analysis (e.g., 2024)")
    max_files: int = Field(default=3, description="Maximum number of XBRL files to download and analyze")


def extract_shareholding_pattern(
    company_query: str,
    from_year: int,
    to_year: int,
    max_files: int = 3
) -> str:
    """
    Extract and analyze shareholding pattern data for a company.
    
    Downloads XBRL files from BSE and provides insights on shareholding patterns,
    promoter holdings, institutional holdings, and foreign investments.
    """
    try:
        # Initialize services
        ingestor = IngestorService()
        analyzer = XBRLAnalyzer()
        
        # Resolve company to BSE scripcode
        scripcode = resolve_bse_scripcode(company_name=company_query)
        if not scripcode:
            return f"❌ **Company Resolution Failed**\n\nCould not resolve '{company_query}' to a valid BSE scripcode. Please try with:\n• Exact company name\n• BSE symbol/ticker\n• 6-digit BSE scripcode"

        # Ingest XBRL data
        print(f"📊 Ingesting XBRL data for scripcode {scripcode} ({from_year}-{to_year})")
        extraction_result = ingestor.extract_shareholding_pattern(
            scripcode=scripcode, 
            from_year=from_year, 
            to_year=to_year, 
            max_xbrl_files=max_files,
            stock_query=company_query
        )
        
        if not extraction_result or extraction_result.get('stored_files_count', 0) == 0:
            return f"❌ **No XBRL Data Found**\n\nNo XBRL files found for scripcode {scripcode} between {from_year}-{to_year}. This could mean:\n• No filings in the specified period\n• Company might be delisted\n• BSE access restrictions"
        
        # Get the directory where files were stored
        stored_files_dir = extraction_result.get('stored_files_dir')
        if not stored_files_dir:
            return f"❌ **Storage Error**\n\nXBRL files were found but could not be stored locally for analysis."
        
        # Analyze the data
        print(f"🔍 Analyzing shareholding patterns from {extraction_result.get('stored_files_count', 0)} files")
        
        # Get list of XBRL files from the directory
        file_paths = []
        if os.path.exists(stored_files_dir):
            for file in os.listdir(stored_files_dir):
                if file.endswith('.xml'):
                    file_paths.append(os.path.join(stored_files_dir, file))
        
        analysis_result = analyzer.analyze_xbrl_files(file_paths, analysis_type="shareholding")
        
        # Format the output using the new storytelling approach
        formatted_response = _format_storytelling_output(
            company_query, 
            from_year, 
            to_year, 
            scripcode, 
            analysis_result,
            extraction_result.get('stored_files_count', 0)
        )
        
        # Clean up temporary files
        stored_files_dir = extraction_result.get('stored_files_dir')
        if stored_files_dir and os.path.exists(stored_files_dir):
            try:
                shutil.rmtree(stored_files_dir)
                logger.info(f"Cleaned up temporary directory: {stored_files_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory {stored_files_dir}: {cleanup_error}")
        
        return formatted_response
        
    except Exception as e:
        logger.exception(f"Error in shareholding pattern extraction: {e}")
        # Attempt cleanup even on error
        try:
            stored_files_dir = extraction_result.get('stored_files_dir') if 'extraction_result' in locals() else None
            if stored_files_dir and os.path.exists(stored_files_dir):
                shutil.rmtree(stored_files_dir)
                logger.info(f"Cleaned up temporary directory after error: {stored_files_dir}")
        except:
            pass
        return f"❌ Error during shareholding pattern analysis: {str(e)}"


def _format_storytelling_output(company_query: str, from_year: int, to_year: int, 
                               scripcode: str, analysis_result, files_count: int) -> str:
    """Format output with storytelling approach, resolving contexts and adding insights"""
    
    if not analysis_result or not analysis_result.shareholding_data:
        return f"❌ **Analysis Failed**\n\nNo shareholding data could be extracted for {company_query} ({from_year}-{to_year})"
    
    # Get analysis insights
    analysis_insights = analysis_result.shareholding_data.get("analysis_insights", {})
    narrative_insights = analysis_insights.get("key_insights", [])
    trends = analysis_insights.get("year_over_year_trends", {})
    breakdown = analysis_insights.get("shareholding_breakdown", {})
    
    # Debug: Log what data we have
    logger.info(f"Analysis insights: {analysis_insights}")
    logger.info(f"Narrative insights: {narrative_insights}")
    logger.info(f"Trends: {trends}")
    logger.info(f"Breakdown: {breakdown}")
    logger.info(f"Full shareholding data keys: {list(analysis_result.shareholding_data.keys())}")
    
    response = []
    
    # Header with filing details
    response.append(f"📈 **OFFICIAL BSE SHAREHOLDING PATTERN - {company_query.upper()}**")
    response.append(f"*Analysis Period: {from_year}-{to_year} | BSE Code: {scripcode} | Files Analyzed: {files_count}*")
    response.append("*Source: BSE XBRL Regulatory Filings*")
    response.append("")
    
    # Key Insights Section (Narrative) - Enhanced with fallback
    if narrative_insights:
        response.append("## 💡 Key Findings from BSE Filings")
        for insight in narrative_insights:
            response.append(f"• {insight}")
        response.append("")
    else:
        # Fallback: Generate insights from raw data if analysis layer failed
        response.append("## 💡 Key Findings from BSE Filings")
        fallback_insights = _generate_fallback_insights(analysis_result.shareholding_data, breakdown, trends)
        for insight in fallback_insights:
            response.append(f"• {insight}")
        response.append("")
    
    # Shareholding Breakdown with storytelling
    if breakdown:
        response.append("## 📊 Official BSE Filing Shareholding Data")
        latest_period = max(breakdown.keys()) if breakdown else None
        
        if latest_period and latest_period in breakdown:
            data = breakdown[latest_period]
            total_identified = sum([data.get(k, 0) for k in ["promoter", "foreign", "institutional", "retail"]])
            
            if total_identified > 0:
                # Create a narrative table with official data
                response.append(f"**Period: {latest_period}**")
                response.append("")
                response.append("| **Category** | **Holding %** | **YoY Change** | **Status** |")
                response.append("|-------------|---------------|----------------|------------|")
                
                for category in ["promoter", "foreign", "institutional", "retail"]:
                    holding = data.get(category, 0)
                    trend_info = trends.get(category, {})
                    
                    if holding > 0:
                        trend_symbol = ""
                        change_text = ""
                        if trend_info.get("change_percentage"):
                            change_pct = trend_info["change_percentage"]
                            if change_pct > 1:
                                trend_symbol = "📈"
                                change_text = f"+{change_pct:.1f}%"
                            elif change_pct < -1:
                                trend_symbol = "📉"
                                change_text = f"{change_pct:.1f}%"
                            else:
                                trend_symbol = "➡️"
                                change_text = "Stable"
                        else:
                            trend_symbol = "📊"
                            change_text = "N/A"
                        
                        category_display = category.replace("_", " ").title()
                        status = "Extracted from XBRL"
                        response.append(f"| {category_display} | {holding:.1f}% | {change_text} | {trend_symbol} {status} |")
                
                response.append("")
                
                # Add shareholder count if available
                shareholder_count = data.get("total_shareholders", 0)
                if shareholder_count > 0:
                    response.append(f"**Total Shareholders (BSE Filing):** {int(shareholder_count):,}")
                    response.append("")
            else:
                # Fallback: Show raw data if computed breakdown is empty
                response.append("### Raw BSE Filing Data")
                raw_patterns = analysis_result.shareholding_data.get('shareholding_patterns', [])
                if raw_patterns:
                    response.append("| **Category** | **Value** | **Source** |")
                    response.append("|-------------|-----------|-----------|")
                    for pattern in raw_patterns[:10]:  # Show top 10 to avoid overflow
                        category = pattern.get('category', 'Unknown')
                        value = pattern.get('value', 'N/A')
                        response.append(f"| {category} | {value} | BSE XBRL |")
                    response.append("")
        else:
            # Show raw patterns if no breakdown available
            response.append("### Available BSE Filing Information")
            raw_patterns = analysis_result.shareholding_data.get('shareholding_patterns', [])
            if raw_patterns:
                response.append("| **Category** | **Details** | **Source** |")
                response.append("|-------------|-------------|-----------|")
                for pattern in raw_patterns[:12]:
                    category = pattern.get('category', 'Unknown')
                    value = pattern.get('value', 'N/A')
                    # Clean up the value display
                    clean_value = str(value).replace('Context', 'Entity').replace('context', 'entity')
                    response.append(f"| {category} | {clean_value} | BSE XBRL |")
                response.append("")
    else:
        # Show raw shareholding patterns if no breakdown available
        response.append("## 📊 BSE Filing Shareholding Information")
        raw_patterns = analysis_result.shareholding_data.get('shareholding_patterns', [])
        if raw_patterns:
            response.append("| **Category** | **Details** | **Source** |")
            response.append("|-------------|-------------|-----------|")
            for pattern in raw_patterns[:15]:  # Limit to prevent overflow
                category = pattern.get('category', 'Unknown')
                value = pattern.get('value', 'N/A')
                # Clean up the value display
                clean_value = str(value).replace('Context', 'Entity').replace('context', 'entity')
                response.append(f"| {category} | {clean_value} | BSE XBRL |")
            response.append("")
    
    # Foreign Investment Story (Enhanced)
    foreign_data = analysis_result.shareholding_data.get('foreign_shareholding_details', [])
    foreign_trend = trends.get("foreign", {})
    
    # Check if we have any foreign-related data in shareholding patterns
    has_foreign_data = (foreign_data or foreign_trend or 
                       any('foreign' in str(p).lower() or 'fpi' in str(p).lower() 
                           for p in analysis_result.shareholding_data.get('shareholding_patterns', [])))
    
    if has_foreign_data:
        response.append("## 🌍 Foreign Investment Data (BSE Filings)")
        
        # Narrative based on trends
        if foreign_trend.get("change_percentage"):
            change_pct = foreign_trend["change_percentage"]
            current_holding = foreign_trend.get("current", 0)
            
            if change_pct > 0:
                response.append(f"**BSE Filing Trend**: Foreign holdings increased by **{change_pct:.1f}%** YoY to **{current_holding:.1f}%**")
            elif change_pct < 0:
                response.append(f"**BSE Filing Trend**: Foreign holdings decreased by **{abs(change_pct):.1f}%** YoY to **{current_holding:.1f}%**")
            else:
                response.append(f"**BSE Filing Status**: Foreign holdings stable at **{current_holding:.1f}%**")
        else:
            # Fallback: Show available foreign data without trends
            foreign_patterns = [p for p in analysis_result.shareholding_data.get('shareholding_patterns', []) 
                              if any(term in p.get('category', '').lower() for term in ['foreign', 'fpi', 'overseas'])]
            if foreign_patterns:
                response.append("**Foreign Investment Entries in BSE Filings**:")
                for pattern in foreign_patterns[:5]:  # Show top 5
                    category = pattern.get('category', 'Unknown')
                    value = pattern.get('value', 'N/A')
                    response.append(f"• **{category}:** {value}")
            elif foreign_data:
                response.append("**Status**: Foreign shareholding details available in BSE regulatory filings")
        
        # List foreign entities from BSE filings
        if foreign_data:
            response.append("")
            response.append("### 🏛️ Foreign Entities (BSE Filing Records)")
            
            resolved_entities = []
            for entity in foreign_data[:10]:  # Show more entities
                name = entity.get('value', 'Unknown')
                original = entity.get('original', '')
                period = entity.get('period', 'N/A')
                
                # Skip if it's still showing context references
                if 'context' not in name.lower() and 'unidentified' not in name.lower():
                    resolved_entities.append(f"• **{name}** *(Period: {period})*")
                elif original and 'context' not in original.lower():
                    resolved_entities.append(f"• **{original}** *(Period: {period})*")
                else:
                    # Clean up context references for better readability
                    clean_name = name.replace('Context', 'Foreign Entity').replace('context', 'foreign entity')
                    if clean_name != name:  # Only add if we made a meaningful change
                        resolved_entities.append(f"• **{clean_name}** *(Period: {period})*")
            
            if resolved_entities:
                response.extend(resolved_entities)
            else:
                response.append("• *Foreign entities identified in filings but require additional name resolution*")
        
        response.append("")
    
    # Major Shareholders (Resolved)
    major_shareholders = analysis_result.shareholding_data.get('major_shareholders', [])
    if major_shareholders:
        response.append("## 💼 Major Stakeholders (BSE Filing Records)")
        
        resolved_shareholders = []
        for shareholder in major_shareholders[:10]:  # Show more stakeholders
            name = shareholder.get('value', 'Unknown')
            original = shareholder.get('original', '')
            period = shareholder.get('period', 'N/A')
            
            # Prefer resolved names over context references
            if 'context' not in name.lower() and 'unidentified' not in name.lower():
                display_name = name
            elif original and 'context' not in original.lower():
                display_name = original
            else:
                # Clean up context references
                display_name = name.replace('Context', 'Stakeholder').replace('context', 'stakeholder')
                if display_name == name:  # If no meaningful change, skip
                    continue
            
            resolved_shareholders.append(f"• **{display_name}** *(Period: {period})*")
        
        if resolved_shareholders:
            response.extend(resolved_shareholders)
        else:
            response.append("• *Major stakeholder information available in BSE filings but requires name resolution*")
        
        response.append("")
    
    # Risk & Opportunity Assessment
    response.append("## ⚖️ BSE Filing Analysis Summary")
    
    risk_opportunities = []
    
    # Assess based on trends and composition
    if trends:
        promoter_trend = trends.get("promoter", {})
        institutional_trend = trends.get("institutional", {})
        foreign_trend = trends.get("foreign", {})
        
        if promoter_trend.get("change_percentage", 0) < -3:
            risk_opportunities.append("⚠️ **BSE Data Alert**: Promoter holdings declined significantly - regulatory filing shows management stake reduction")
        elif promoter_trend.get("change_percentage", 0) > 3:
            risk_opportunities.append("🚀 **BSE Data Positive**: Promoter holdings increased - regulatory filing shows strengthened management commitment")
        
        if institutional_trend.get("change_percentage", 0) > 5:
            risk_opportunities.append("🚀 **BSE Data Positive**: Institutional participation surged - regulatory filing shows professional investor confidence")
        elif institutional_trend.get("change_percentage", 0) < -5:
            risk_opportunities.append("⚠️ **BSE Data Alert**: Institutional holdings declined - worth monitoring for sentiment shift")
        
        if foreign_trend.get("change_percentage", 0) > 3:
            risk_opportunities.append("🌟 **BSE Data Positive**: Foreign investment increased - regulatory filing shows global investor interest")
        elif foreign_trend.get("change_percentage", 0) < -3:
            risk_opportunities.append("⚠️ **BSE Data Alert**: Foreign holdings decreased - may indicate global portfolio rebalancing")
    
    # Assess based on raw data if no trends available
    if not risk_opportunities:
        raw_patterns = analysis_result.shareholding_data.get('shareholding_patterns', [])
        total_patterns = len(raw_patterns)
        
        if total_patterns > 15:
            risk_opportunities.append("📊 **Data Quality**: Comprehensive BSE filing with detailed shareholding disclosure")
        elif total_patterns > 8:
            risk_opportunities.append("📊 **Data Quality**: Good level of shareholding transparency in BSE regulatory filings")
        elif total_patterns > 0:
            risk_opportunities.append("📊 **Data Quality**: Basic shareholding information available in BSE filings")
        
        # Check for foreign participation
        foreign_patterns = [p for p in raw_patterns if any(term in p.get('category', '').lower() 
                           for term in ['foreign', 'fpi', 'overseas'])]
        if foreign_patterns:
            risk_opportunities.append("🌍 **Global Interest**: Foreign investor entries found in BSE regulatory filings")
        
        # Check for institutional participation
        institutional_patterns = [p for p in raw_patterns if any(term in p.get('category', '').lower() 
                                 for term in ['institution', 'mutual', 'insurance'])]
        if institutional_patterns:
            risk_opportunities.append("🏦 **Institutional Base**: Institutional investor presence confirmed in BSE filings")
    
    # Default assessment if no specific insights
    if not risk_opportunities:
        risk_opportunities.append("📊 **Status**: BSE shareholding data successfully extracted from regulatory filings")
    
    # Add data source disclaimer
    response.extend(risk_opportunities)
    response.append("")
    response.append("**Note**: This analysis is based on official BSE XBRL regulatory filings. For comprehensive market analysis with detailed entity names and current market data, additional research tools will be used to enhance these findings.")
    
    return "\n".join(response)


def _generate_fallback_insights(shareholding_data: Dict, breakdown: Dict, trends: Dict) -> List[str]:
    """Generate fallback insights when analysis layer doesn't provide them"""
    insights = []
    
    # Extract raw shareholding patterns if available
    raw_patterns = shareholding_data.get('shareholding_patterns', [])
    foreign_details = shareholding_data.get('foreign_shareholding_details', [])
    major_shareholders = shareholding_data.get('major_shareholders', [])
    
    if raw_patterns:
        insights.append(f"Successfully extracted {len(raw_patterns)} shareholding data points from XBRL filings")
        
        # Look for specific patterns
        promoter_data = [p for p in raw_patterns if 'promot' in p.get('category', '').lower()]
        foreign_data = [p for p in raw_patterns if any(term in p.get('category', '').lower() for term in ['foreign', 'fpi'])]
        institutional_data = [p for p in raw_patterns if any(term in p.get('category', '').lower() for term in ['institution', 'mutual'])]
        
        if promoter_data:
            insights.append(f"Promoter shareholding information identified across {len(promoter_data)} categories")
        if foreign_data:
            insights.append(f"Foreign investment details found across {len(foreign_data)} classifications")
        if institutional_data:
            insights.append(f"Institutional holdings data available across {len(institutional_data)} categories")
    
    if foreign_details:
        insights.append(f"Detailed foreign shareholding information available for {len(foreign_details)} entities")
    
    if major_shareholders:
        insights.append(f"Major stakeholder information identified for {len(major_shareholders)} key participants")
    
    # If we have breakdown data, provide specific insights
    if breakdown:
        latest_period = max(breakdown.keys()) if breakdown else None
        if latest_period and latest_period in breakdown:
            period_data = breakdown[latest_period]
            total_holdings = sum([period_data.get(k, 0) for k in ['promoter', 'foreign', 'institutional', 'retail']])
            if total_holdings > 50:  # Reasonable threshold for meaningful data
                insights.append(f"Comprehensive shareholding breakdown available with {total_holdings:.1f}% coverage")
    
    # Fallback if no specific insights
    if not insights:
        insights.append("Shareholding pattern data extracted and processed from XBRL filings")
        insights.append("Detailed analysis available upon request for specific metrics")
    
    return insights


def _cleanup_files(directory: str, analyzer: XBRLAnalyzer) -> None:
    """Helper function to clean up temporary files immediately"""
    try:
        if os.path.exists(directory):
            # Remove the entire directory and all its contents
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary directory: {directory}")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")


# Create structured tools for LangGraph
shareholding_pattern_tool = StructuredTool.from_function(
    func=extract_shareholding_pattern,
    name="extract_shareholding_pattern",
    description=(
        "Extract and analyze shareholding pattern data for an Indian company. "
        "Downloads XBRL files from BSE and provides detailed insights on promoter holdings, "
        "institutional investments, foreign holdings, and ownership structure changes over time. "
        "Provides key metrics and trends in shareholding patterns."
    ),
    args_schema=ShareholdingPatternInput,
)
