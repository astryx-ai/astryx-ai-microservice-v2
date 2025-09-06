"""
LangGraph tools for financial data extraction and XBRL analysis.
"""
import os
import shutil
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from app.services.report import IngestorService
from app.agent_tools.xbrl_analyzer import XBRLAnalyzer
from app.db.companies import resolve_bse_scripcode
from app.agent_tools.formatter import format_financial_content

logger = logging.getLogger("financial_tools")


class ShareholdingPatternInput(BaseModel):
    """Input schema for shareholding pattern extraction"""
    company_query: str = Field(..., description="Company name, ticker symbol, or BSE code to analyze")
    from_year: int = Field(..., description="Start year for analysis (e.g., 2022)")
    to_year: int = Field(..., description="End year for analysis (e.g., 2024)")
    max_files: int = Field(default=3, description="Maximum number of XBRL files to download and analyze")


class FundamentalsHeaderInput(BaseModel):
    """Input schema for fundamentals header extraction"""
    company_query: str = Field(..., description="Company name, ticker symbol, or BSE code to analyze")
    quote_type: str = Field(default="EQ", description="Quote type - typically 'EQ' for equity")


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
        logger.info(f"📊 Ingesting XBRL data for scripcode {scripcode} ({from_year}-{to_year})")
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
        logger.info(f"🔍 Analyzing shareholding patterns from {extraction_result.get('stored_files_count', 0)} files")
        
        # Get list of XBRL files from the directory
        file_paths = []
        if os.path.exists(stored_files_dir):
            for file in os.listdir(stored_files_dir):
                if file.endswith('.xml'):
                    file_paths.append(os.path.join(stored_files_dir, file))
        
        analysis_result = analyzer.analyze_xbrl_files(file_paths, analysis_type="shareholding")
        
        # Format the output
        raw_response = _format_analysis_output(
            company_query, 
            from_year, 
            to_year, 
            scripcode, 
            analysis_result,
            extraction_result.get('stored_files_count', 0)
        )
        
        # Use the formatter to clean and structure the response
        formatted_response = format_financial_content(raw_response)
        
        # Clean up temporary files
        _cleanup_temp_files(stored_files_dir)
        
        return formatted_response
        
    except Exception as e:
        logger.exception(f"Error in shareholding pattern extraction: {e}")
        # Attempt cleanup even on error
        if 'stored_files_dir' in locals() and stored_files_dir:
            _cleanup_temp_files(stored_files_dir)
        return f"❌ Error during shareholding pattern analysis: {str(e)}"


def extract_fundamentals_header(
    company_query: str,
    quote_type: str = "EQ"
) -> str:
    """
    Extract fundamental/company header data from BSE API.
    
    Retrieves basic company information, current market data, and fundamental 
    metrics like market cap, PE ratio, book value, etc.
    """
    try:
        # Initialize service
        ingestor = IngestorService()
        
        # Resolve company to BSE scripcode
        scripcode = resolve_bse_scripcode(company_name=company_query)
        if not scripcode:
            return f"❌ **Company Resolution Failed**\n\nCould not resolve '{company_query}' to a valid BSE scripcode. Please try with:\n• Exact company name\n• BSE symbol/ticker\n• 6-digit BSE scripcode"

        # Extract fundamentals header data
        logger.info(f"📈 Fetching fundamentals header for scripcode {scripcode}")
        
        extraction_result = ingestor.extract_fundamentals_header(
            scripcode=scripcode,
            stock_query=company_query,
            quote_type=quote_type
        )
        
        if extraction_result.get('status') != 'success':
            error_msg = extraction_result.get('error', 'Unknown error occurred')
            return f"❌ **Data Extraction Failed**\n\n{error_msg}"
        
        # Format the response
        raw_response = extraction_result.get('raw_response', {})
        formatted_response = _format_fundamentals_output(
            company_query,
            scripcode,
            quote_type,
            raw_response
        )
        
        # Use the formatter to clean and structure the response
        final_response = format_financial_content(formatted_response)
        
        return final_response
        
    except Exception as e:
        logger.exception(f"Error in fundamentals header extraction: {e}")
        return f"❌ Error during fundamentals header extraction: {str(e)}"


def _format_fundamentals_output(company_query: str, scripcode: str, quote_type: str, raw_data: Dict[str, Any]) -> str:
    """Format fundamentals header output with clean structure - only show available data"""
    
    if not raw_data:
        return f"❌ **No Data Found**\n\nNo fundamentals data could be retrieved for {company_query}"
    
    # Count available vs total fields to determine data quality
    available_fields = sum(1 for v in raw_data.values() if v and str(v).strip() and str(v) not in ['None', 'null', ''])
    total_fields = len(raw_data)
    
    response = []
    
    # Header with data availability insight
    response.append(f"# BSE Fundamentals - {company_query.upper()}")
    response.append(f"**BSE Code:** {scripcode} | **Data Fields Available:** {available_fields}/{total_fields}")
    response.append("**Source:** BSE ComHeadernew API")
    response.append("")
    
    # Helper function to safely get and format field
    def get_field(key, prefix="", suffix=""):
        value = raw_data.get(key)
        if value and str(value).strip() and str(value) not in ['None', 'null', '']:
            return f"{prefix}{value}{suffix}"
        return None
    
    # Only show sections with available data
    sections_added = 0
    
    # Company Information - only if we have data
    company_fields = []
    security_id = get_field('SecurityId')
    if security_id:
        company_fields.append(f"**Security ID:** {security_id}")
    
    isin = get_field('ISIN')
    if isin:
        company_fields.append(f"**ISIN:** {isin}")
        
    industry = get_field('Industry')
    if industry:
        company_fields.append(f"**Industry:** {industry}")
    
    sector = get_field('Sector')
    if sector:
        company_fields.append(f"**Sector:** {sector}")
        
    igroup = get_field('IGroup')
    if igroup:
        company_fields.append(f"**Group:** {igroup}")
        
    isubgroup = get_field('ISubGroup')
    if isubgroup:
        company_fields.append(f"**Sub-Group:** {isubgroup}")
    
    if company_fields:
        response.append("## Company Information")
        response.extend(company_fields)
        response.append("")
        sections_added += 1
    
    # Financial Ratios - only if we have data
    ratio_fields = []
    pe_ratio = get_field('PE')
    if pe_ratio:
        ratio_fields.append(f"**P/E Ratio:** {pe_ratio}")
        
    pb_ratio = get_field('PB')
    if pb_ratio:
        ratio_fields.append(f"**P/B Ratio:** {pb_ratio}")
        
    roe = get_field('ROE', suffix="%")
    if roe:
        ratio_fields.append(f"**ROE:** {roe}")
        
    npm = get_field('NPM', suffix="%")
    if npm:
        ratio_fields.append(f"**Net Profit Margin:** {npm}")
        
    eps = get_field('EPS', prefix="₹")
    if eps:
        ratio_fields.append(f"**EPS:** {eps}")
    
    if ratio_fields:
        response.append("## Key Financial Ratios")
        response.extend(ratio_fields)
        response.append("")
        sections_added += 1
    
    # Basic Details
    basic_fields = []
    face_val = get_field('FaceVal', prefix="₹")
    if face_val:
        basic_fields.append(f"**Face Value:** {face_val}")
        
    group = get_field('Group')
    if group:
        basic_fields.append(f"**BSE Group:** {group}")
        
    index = get_field('Index')
    if index:
        basic_fields.append(f"**Index:** {index}")
        
    settl_type = get_field('SetlType')
    if settl_type:
        basic_fields.append(f"**Settlement:** {settl_type}")
    
    if basic_fields:
        response.append("## Basic Details")
        response.extend(basic_fields)
        response.append("")
        sections_added += 1
    
    # Only show error message if no data sections were added
    if sections_added == 0:
        response.append("❌ **Limited Data Available**")
        response.append(f"Only {available_fields} fields returned from BSE API, with no key financial metrics.")
    
    return "\n".join(response)


def _cleanup_temp_files(directory: str) -> None:
    """Clean up temporary files"""
    if directory and os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary directory: {directory}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp directory {directory}: {cleanup_error}")


def _format_analysis_output(company_query: str, from_year: int, to_year: int, 
                           scripcode: str, analysis_result, files_count: int) -> str:
    """Format analysis output with clean structure"""
    
    if not analysis_result or not analysis_result.shareholding_data:
        return f"❌ **Analysis Failed**\n\nNo shareholding data could be extracted for {company_query} ({from_year}-{to_year})"
    
    # Get analysis data
    shareholding_data = analysis_result.shareholding_data
    analysis_insights = shareholding_data.get("analysis_insights", {})
    narrative_insights = analysis_insights.get("key_insights", [])
    trends = analysis_insights.get("year_over_year_trends", {})
    breakdown = analysis_insights.get("shareholding_breakdown", {})
    
    response = []
    
    # Header
    response.append(f"# BSE Shareholding Pattern Analysis - {company_query.upper()}")
    response.append(f"**Analysis Period:** {from_year}-{to_year} | **BSE Code:** {scripcode} | **Files Analyzed:** {files_count}")
    response.append("**Source:** BSE XBRL Regulatory Filings")
    response.append("")
    
    # Executive Summary
    response.append("## Executive Summary")
    
    if narrative_insights:
        for insight in narrative_insights:
            response.append(f"• {insight}")
    else:
        # Generate fallback insights
        fallback_insights = _generate_insights(shareholding_data)
        for insight in fallback_insights:
            response.append(f"• {insight}")
    response.append("")
    
    # Shareholding Breakdown
    response.append("## Shareholding Composition")
    
    if breakdown:
        latest_period = max(breakdown.keys()) if breakdown else None
        if latest_period and latest_period in breakdown:
            data = breakdown[latest_period]
            _add_breakdown_table(response, data, trends, latest_period)
        else:
            _add_raw_patterns_table(response, shareholding_data)
    else:
        _add_raw_patterns_table(response, shareholding_data)
    
    # Foreign Investment Analysis
    _add_foreign_investment_analysis(response, shareholding_data, trends)
    
    # Major Shareholders
    _add_major_shareholders_analysis(response, shareholding_data)
    
    # Key Insights & Assessment
    _add_assessment_section(response, trends, shareholding_data)
    
    # Disclaimer
    response.append("")
    response.append("---")
    response.append("**Note:** This analysis is based on official BSE XBRL regulatory filings. For comprehensive market analysis with current market data, additional research tools may be utilized.")
    
    return "\n".join(response)


def _generate_insights(shareholding_data: Dict) -> List[str]:
    """Generate insights from shareholding data"""
    insights = []
    
    raw_patterns = shareholding_data.get('shareholding_patterns', [])
    foreign_details = shareholding_data.get('foreign_shareholding_details', [])
    major_shareholders = shareholding_data.get('major_shareholders', [])
    
    if raw_patterns:
        insights.append(f"Successfully extracted {len(raw_patterns)} shareholding data points from XBRL filings")
        
        # Categorize patterns
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
    
    if not insights:
        insights.append("Shareholding pattern data extracted and processed from XBRL filings")
    
    return insights


def _add_breakdown_table(response: List[str], data: Dict, trends: Dict, period: str) -> None:
    """Add shareholding breakdown table"""
    response.append(f"**Period:** {period}")
    response.append("")
    response.append("| Category | Holding % | YoY Change | Status |")
    response.append("|----------|-----------|------------|--------|")
    
    for category in ["promoter", "foreign", "institutional", "retail"]:
        holding = data.get(category, 0)
        if holding > 0:
            trend_info = trends.get(category, {})
            change_text = "N/A"
            
            if trend_info.get("change_percentage"):
                change_pct = trend_info["change_percentage"]
                if change_pct > 1:
                    change_text = f"+{change_pct:.1f}%"
                elif change_pct < -1:
                    change_text = f"{change_pct:.1f}%"
                else:
                    change_text = "Stable"
            
            category_name = category.replace("_", " ").title()
            response.append(f"| {category_name} | {holding:.1f}% | {change_text} | XBRL Data |")
    
    response.append("")


def _add_raw_patterns_table(response: List[str], shareholding_data: Dict) -> None:
    """Add raw patterns table when breakdown is not available"""
    raw_patterns = shareholding_data.get('shareholding_patterns', [])
    if raw_patterns:
        response.append("| Category | Details | Source |")
        response.append("|----------|---------|--------|")
        for pattern in raw_patterns[:10]:
            category = pattern.get('category', 'Unknown')
            value = pattern.get('value', 'N/A')
            clean_value = str(value).replace('Context', 'Entity').replace('context', 'entity')[:50]
            response.append(f"| {category} | {clean_value} | BSE XBRL |")
        response.append("")


def _add_foreign_investment_analysis(response: List[str], shareholding_data: Dict, trends: Dict) -> None:
    """Add foreign investment analysis section"""
    foreign_data = shareholding_data.get('foreign_shareholding_details', [])
    foreign_trend = trends.get("foreign", {})
    
    # Check if we have foreign-related data
    has_foreign_data = foreign_data or foreign_trend or any(
        'foreign' in str(p).lower() or 'fpi' in str(p).lower() 
        for p in shareholding_data.get('shareholding_patterns', [])
    )
    
    if has_foreign_data:
        response.append("## Foreign Investment Analysis")
        
        # Add trend information
        if foreign_trend.get("change_percentage"):
            change_pct = foreign_trend["change_percentage"]
            current_holding = foreign_trend.get("current", 0)
            
            if change_pct > 0:
                response.append(f"**Trend:** Foreign holdings increased by {change_pct:.1f}% YoY to {current_holding:.1f}%")
            elif change_pct < 0:
                response.append(f"**Trend:** Foreign holdings decreased by {abs(change_pct):.1f}% YoY to {current_holding:.1f}%")
            else:
                response.append(f"**Status:** Foreign holdings stable at {current_holding:.1f}%")
        else:
            response.append("**Status:** Foreign investment patterns identified in BSE regulatory filings")
        
        # List foreign entities
        if foreign_data:
            response.append("")
            response.append("**Foreign Entities (from BSE filings):**")
            for entity in foreign_data[:5]:
                name = entity.get('value', 'Unknown')
                if 'context' not in name.lower() and len(name) > 3:
                    response.append(f"• {name}")
        
        response.append("")


def _add_major_shareholders_analysis(response: List[str], shareholding_data: Dict) -> None:
    """Add major shareholders analysis section"""
    major_shareholders = shareholding_data.get('major_shareholders', [])
    if major_shareholders:
        response.append("## Major Stakeholders")
        
        valid_shareholders = []
        for shareholder in major_shareholders[:8]:
            name = shareholder.get('value', 'Unknown')
            if 'context' not in name.lower() and len(name) > 3:
                valid_shareholders.append(f"• {name}")
        
        if valid_shareholders:
            response.extend(valid_shareholders)
        else:
            response.append("• Major stakeholder information available in regulatory filings")
        
        response.append("")


def _add_assessment_section(response: List[str], trends: Dict, shareholding_data: Dict) -> None:
    """Add assessment and key insights section"""
    response.append("## Key Assessment")
    
    assessments = []
    
    # Trend-based assessments
    if trends:
        promoter_trend = trends.get("promoter", {})
        institutional_trend = trends.get("institutional", {})
        foreign_trend = trends.get("foreign", {})
        
        if promoter_trend.get("change_percentage", 0) < -3:
            assessments.append("⚠️ Declining promoter holdings may indicate reduced management confidence")
        elif promoter_trend.get("change_percentage", 0) > 3:
            assessments.append("✅ Increasing promoter holdings indicates strong management confidence")
        
        if institutional_trend.get("change_percentage", 0) > 5:
            assessments.append("📈 Growing institutional interest suggests professional validation")
        elif institutional_trend.get("change_percentage", 0) < -5:
            assessments.append("⚠️ Declining institutional holdings may indicate reduced confidence")
        
        if foreign_trend.get("change_percentage", 0) > 3:
            assessments.append("🌐 Increasing foreign investment indicates international appeal")
        elif foreign_trend.get("change_percentage", 0) < -3:
            assessments.append("📉 Declining foreign investment may reflect global sentiment changes")
    
    # Data-based assessments
    if not assessments:
        raw_patterns = shareholding_data.get('shareholding_patterns', [])
        total_patterns = len(raw_patterns)
        
        if total_patterns > 15:
            assessments.append("📊 Comprehensive shareholding structure with detailed stakeholder information")
        elif total_patterns > 8:
            assessments.append("📊 Standard shareholding disclosure with key stakeholder data")
        elif total_patterns > 0:
            assessments.append("📊 Basic shareholding data available from regulatory filings")
        else:
            assessments.append("📊 BSE shareholding data successfully extracted from regulatory filings")
    
    # Add assessments
    for assessment in assessments:
        response.append(f"• {assessment}")


# Create structured tool for LangGraph
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

# Create structured tool for fundamentals header
fundamentals_header_tool = StructuredTool.from_function(
    func=extract_fundamentals_header,
    name="extract_fundamentals_header",
    description=(
        "🏦 Extract key fundamental ratios and company data from BSE API for Indian listed companies. "
        "Returns available data including P/E ratio, P/B ratio, ROE, EPS, industry classification, "
        "and company details. Use this when users ask for 'fundamentals', 'P/E ratio', 'company info', "
        "or financial ratios. Note: This API provides fundamental ratios but NOT live market prices, "
        "current trading data, or market cap. Only shows data that is actually available (no N/A fields)."
    ),
    args_schema=FundamentalsHeaderInput,
)
