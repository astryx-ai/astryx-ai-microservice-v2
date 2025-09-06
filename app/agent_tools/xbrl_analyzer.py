"""
XBRL Analysis Tools for processing financial data from XBRL files.
"""
import os
import re
import json
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("xbrl_analyzer")


class XBRLAnalysisResult(BaseModel):
    """Result structure for XBRL analysis"""
    success: bool = Field(..., description="Whether analysis was successful")
    summary: str = Field(..., description="Summary of key insights")
    financial_data: Dict[str, Any] = Field(default_factory=dict, description="Extracted financial data")
    shareholding_data: Dict[str, Any] = Field(default_factory=dict, description="Shareholding pattern data")
    governance_data: Dict[str, Any] = Field(default_factory=dict, description="Corporate governance data")
    key_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Key financial metrics")
    insights: List[str] = Field(default_factory=list, description="Key insights and observations")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class XBRLAnalyzer:
    """Analyzer for XBRL financial data files"""
    
    def __init__(self):
        self.temp_cleanup_age = timedelta(minutes=30)  # Cleanup temp files after 30 minutes
        self.entity_mapping = {}  # Store context ID to entity name mappings
    
    def analyze_xbrl_files(self, file_paths: List[str], analysis_type: str = "shareholding") -> XBRLAnalysisResult:
        """
        Analyze XBRL files and extract key insights
        
        Args:
            file_paths: List of file paths to XBRL files
            analysis_type: Type of analysis ("shareholding" or "governance")
        """
        result = XBRLAnalysisResult(
            success=False,
            summary="",
            financial_data={},
            shareholding_data={},
            governance_data={},
            key_metrics=[],
            insights=[],
            errors=[]
        )
        
        if not file_paths:
            result.errors.append("No XBRL files provided for analysis")
            return result
        
        try:
            # Process each file
            all_data = []
            for file_path in file_paths:
                try:
                    if os.path.exists(file_path):
                        data = self._parse_xbrl_file(file_path, analysis_type)
                        if data:
                            all_data.append(data)
                    else:
                        result.errors.append(f"File not found: {file_path}")
                except Exception as e:
                    result.errors.append(f"Error processing {file_path}: {str(e)}")
            
            if not all_data:
                result.errors.append("No valid data extracted from XBRL files")
                return result
            
            # Analyze the extracted data
            if analysis_type == "shareholding":
                result = self._analyze_shareholding_data(all_data, result)
            elif analysis_type == "governance":
                result = self._analyze_governance_data(all_data, result)
            else:
                result.errors.append(f"Unknown analysis type: {analysis_type}")
                return result
            
            result.success = True
            
        except Exception as e:
            result.errors.append(f"Analysis failed: {str(e)}")
            logger.exception("XBRL analysis failed")
        
        return result
    
    def _parse_xbrl_file(self, file_path: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Parse a single XBRL file and extract relevant data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as XML
            try:
                root = ET.fromstring(content)
                return self._extract_xml_data(root, analysis_type)
            except ET.ParseError:
                # If XML parsing fails, try text-based extraction
                return self._extract_text_data(content, analysis_type)
                
        except Exception as e:
            logger.error(f"Failed to parse XBRL file {file_path}: {e}")
            return None
    
    def _extract_xml_data(self, root: ET.Element, analysis_type: str) -> Dict[str, Any]:
        """Extract data from parsed XML"""
        data = {
            "type": analysis_type,
            "financial_items": {},
            "metadata": {},
            "raw_elements": [],
            "entity_mapping": {}  # Store context to entity mappings
        }
        
        # Extract entity mappings first
        data["entity_mapping"] = self._extract_entity_mappings(root)
        
        # Extract basic metadata
        data["metadata"]["filing_date"] = self._find_element_text(root, ["FilingDate", "ReportingDate", "PeriodEndDate"])
        data["metadata"]["company_name"] = self._find_element_text(root, ["CompanyName", "EntityName", "NameOfCompany"])
        data["metadata"]["period"] = self._find_element_text(root, ["Period", "ReportingPeriod", "Quarter"])
        
        if analysis_type == "shareholding":
            # Extract shareholding-specific data
            shareholding_patterns = []
            
            # Look for common shareholding elements
            for elem in root.iter():
                tag_lower = elem.tag.lower() if elem.tag else ""
                text = elem.text or ""
                
                if any(keyword in tag_lower for keyword in ['sharehold', 'equity', 'promot', 'foreign', 'institutional']):
                    if text.strip() and not text.strip().isspace():
                        # Resolve context references
                        resolved_value = self._resolve_context_reference(text, data["entity_mapping"])
                        shareholding_patterns.append({
                            "category": elem.tag,
                            "value": resolved_value,
                            "original_value": text.strip(),
                            "attributes": elem.attrib
                        })
            
            data["shareholding_patterns"] = shareholding_patterns
            
        elif analysis_type == "governance":
            # Extract governance-specific data
            governance_items = []
            
            for elem in root.iter():
                tag_lower = elem.tag.lower() if elem.tag else ""
                text = elem.text or ""
                
                if any(keyword in tag_lower for keyword in ['director', 'board', 'committee', 'audit', 'compliance', 'governance']):
                    if text.strip() and not text.strip().isspace():
                        governance_items.append({
                            "category": elem.tag,
                            "value": text.strip(),
                            "attributes": elem.attrib
                        })
            
            data["governance_items"] = governance_items
        
        return data
    
    def _extract_text_data(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Extract data using text-based patterns when XML parsing fails"""
        data = {
            "type": analysis_type,
            "extracted_values": {},
            "raw_content": content[:1000],  # First 1000 chars for context
            "patterns_found": []
        }
        
        if analysis_type == "shareholding":
            # Look for shareholding patterns
            patterns = {
                "promoter_percentage": r"promoter[s]?\s*[:=]\s*([0-9.]+)%?",
                "public_percentage": r"public\s*[:=]\s*([0-9.]+)%?",
                "foreign_percentage": r"foreign\s*[:=]\s*([0-9.]+)%?",
                "institutional_percentage": r"institutional\s*[:=]\s*([0-9.]+)%?"
            }
            
            for key, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    data["extracted_values"][key] = matches[0]
                    data["patterns_found"].append(key)
        
        elif analysis_type == "governance":
            # Look for governance patterns
            patterns = {
                "board_size": r"board\s+size\s*[:=]\s*([0-9]+)",
                "independent_directors": r"independent\s+director[s]?\s*[:=]\s*([0-9]+)",
                "audit_committee": r"audit\s+committee\s+member[s]?\s*[:=]\s*([0-9]+)",
                "women_directors": r"women\s+director[s]?\s*[:=]\s*([0-9]+)"
            }
            
            for key, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    data["extracted_values"][key] = matches[0]
                    data["patterns_found"].append(key)
        
        return data
    
    def _find_element_text(self, root: ET.Element, tag_names: List[str]) -> Optional[str]:
        """Find text content for any of the given tag names"""
        for tag_name in tag_names:
            for elem in root.iter():
                if elem.tag and tag_name.lower() in elem.tag.lower():
                    if elem.text and elem.text.strip():
                        return elem.text.strip()
        return None
    
    def _analyze_shareholding_data(self, all_data: List[Dict[str, Any]], result: XBRLAnalysisResult) -> XBRLAnalysisResult:
        """Analyze shareholding pattern data and generate insights"""
        shareholding_summary = {}
        foreign_shareholders = []
        top_shareholders = []
        
        # Aggregate entity mappings from all files
        combined_entity_mapping = {}
        for data in all_data:
            if "entity_mapping" in data:
                combined_entity_mapping.update(data["entity_mapping"])
        
        # Aggregate data from all files with entity resolution
        for data in all_data:
            if "shareholding_patterns" in data:
                for pattern in data["shareholding_patterns"]:
                    category = pattern["category"].lower()
                    value = pattern["value"]  # This should already be resolved
                    original_value = pattern.get("original_value", value)
                    
                    # Try to extract numerical values
                    numbers = re.findall(r'([0-9.]+)', str(value))
                    if numbers:
                        shareholding_summary[category] = {
                            "value": numbers[0],
                            "text": value,
                            "original": original_value,
                            "period": data.get("metadata", {}).get("period", "Unknown")
                        }
                    
                    # Extract foreign shareholder details
                    if any(foreign_term in category for foreign_term in ['foreign', 'fpi', 'overseas', 'international']):
                        foreign_shareholders.append({
                            "category": pattern["category"],
                            "value": value,
                            "original": original_value,
                            "period": data.get("metadata", {}).get("period", "Unknown")
                        })
                    
                    # Extract potential shareholder names and details
                    if any(term in category for term in ['shareholder', 'investor', 'institution', 'fund']):
                        top_shareholders.append({
                            "name": pattern["category"],
                            "value": value,
                            "original": original_value,
                            "period": data.get("metadata", {}).get("period", "Unknown")
                        })
            
            if "extracted_values" in data:
                shareholding_summary.update(data["extracted_values"])
        
        # Add analysis layer
        analysis_insights = self._add_analysis_layer(all_data, result)
        
        # Store data in result
        result.shareholding_data = shareholding_summary
        result.shareholding_data["foreign_shareholding_details"] = foreign_shareholders
        result.shareholding_data["major_shareholders"] = top_shareholders
        result.shareholding_data["analysis_insights"] = analysis_insights
        
        # Enhanced insights from analysis layer
        key_insights = analysis_insights.get("key_insights", [])
        
        if shareholding_summary:
            if not key_insights:  # Fallback if no analytical insights
                key_insights.append("✅ Shareholding pattern data successfully extracted from XBRL filings")
        
        result.insights = key_insights
        result.summary = (
            f"📊 Analyzed {len(all_data)} XBRL files for shareholding patterns. "
            f"Generated {len(key_insights)} narrative insights with trend analysis."
        )
        
        return result
    
    def _categorize_shareholding_metric(self, metric_name: str) -> str:
        """Categorize shareholding metrics for better organization"""
        metric_lower = metric_name.lower()
        
        if any(term in metric_lower for term in ['promot', 'director', 'management']):
            return "Promoter Holdings"
        elif any(term in metric_lower for term in ['foreign', 'fpi', 'overseas', 'international']):
            return "Foreign Holdings"
        elif any(term in metric_lower for term in ['institution', 'mutual', 'insurance', 'bank']):
            return "Institutional Holdings"
        elif any(term in metric_lower for term in ['public', 'retail', 'individual']):
            return "Public Holdings"
        else:
            return "Other Holdings"
    
    def _analyze_governance_data(self, all_data: List[Dict[str, Any]], result: XBRLAnalysisResult) -> XBRLAnalysisResult:
        """Analyze corporate governance data and generate insights"""
        governance_summary = {}
        key_insights = []
        
        # Aggregate data from all files
        for data in all_data:
            if "governance_items" in data:
                for item in data["governance_items"]:
                    category = item["category"].lower()
                    value = item["value"]
                    
                    # Try to extract numerical values
                    numbers = re.findall(r'([0-9.]+)', value)
                    if numbers:
                        governance_summary[category] = {
                            "value": numbers[0],
                            "text": value,
                            "period": data.get("metadata", {}).get("period", "Unknown")
                        }
            
            if "extracted_values" in data:
                governance_summary.update(data["extracted_values"])
        
        result.governance_data = governance_summary
        
        # Generate insights
        if governance_summary:
            key_insights.append("Corporate governance data successfully extracted from XBRL files")
            
            # Look for board composition
            board_keys = [k for k in governance_summary.keys() if any(term in k.lower() for term in ['board', 'director'])]
            if board_keys:
                key_insights.append(f"Board composition information found: {len(board_keys)} entries")
            
            # Look for committee information
            committee_keys = [k for k in governance_summary.keys() if 'committee' in k.lower()]
            if committee_keys:
                key_insights.append(f"Committee information found: {len(committee_keys)} entries")
            
            # Generate key metrics
            result.key_metrics = [
                {"metric": k, "value": v.get("value", v) if isinstance(v, dict) else v}
                for k, v in governance_summary.items()
                if isinstance(v, (str, int, float, dict))
            ]
        
        result.insights = key_insights
        result.summary = f"Analyzed {len(all_data)} XBRL files for corporate governance. " + \
                        f"Extracted {len(governance_summary)} data points. " + \
                        f"Key insights: {len(key_insights)} findings."
        
        return result
    
    def cleanup_temp_files(self, base_dir: str) -> int:
        """Clean up temporary files older than cleanup_age"""
        if not os.path.exists(base_dir):
            return 0
        
        cleaned_count = 0
        cutoff_time = datetime.now() - self.temp_cleanup_age
        
        try:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
            
            # Remove empty directories
            for root, dirs, files in os.walk(base_dir, topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count
    
    def _extract_entity_mappings(self, root: ET.Element) -> Dict[str, str]:
        """Extract context ID to entity name mappings from XBRL"""
        mappings = {}
        
        # Look for context definitions
        for elem in root.iter():
            tag_lower = elem.tag.lower() if elem.tag else ""
            
            # Common patterns for entity mappings in XBRL
            if 'context' in tag_lower or 'entity' in tag_lower:
                context_id = elem.get('id', '')
                entity_name = elem.text or ''
                
                # Also check attributes for entity info
                for attr_name, attr_value in elem.attrib.items():
                    if 'name' in attr_name.lower() or 'entity' in attr_name.lower():
                        entity_name = attr_value
                        break
                
                if context_id and entity_name:
                    mappings[context_id] = entity_name
            
            # Look for explicit shareholder names
            if any(term in tag_lower for term in ['shareholder', 'investor', 'institution', 'name']):
                text = elem.text or ''
                if text and not any(skip in text.lower() for skip in ['context', 'total', 'number']):
                    # Try to find associated context ID
                    context_ref = elem.get('contextRef', '') or elem.get('context', '')
                    if context_ref:
                        mappings[context_ref] = text
        
        return mappings
    
    def _resolve_context_reference(self, value: str, entity_mapping: Dict[str, str]) -> str:
        """Resolve context references to actual entity names"""
        if not value or not isinstance(value, str):
            return value
        
        # Check if value looks like a context reference (e.g., Context15, c_123, etc.)
        context_pattern = re.match(r'^(context|c_?)(\d+)$', value.lower().strip())
        if context_pattern:
            # Look for mapping
            if value in entity_mapping:
                return entity_mapping[value]
            else:
                return f"Unidentified shareholder ({value})"
        
        # Check if value contains context references mixed with other text
        context_refs = re.findall(r'\b(context\d+|c_\d+)\b', value.lower())
        if context_refs:
            resolved_value = value
            for ref in context_refs:
                if ref in entity_mapping:
                    resolved_value = resolved_value.replace(ref, entity_mapping[ref])
                else:
                    resolved_value = resolved_value.replace(ref, f"Unidentified shareholder ({ref})")
            return resolved_value
        
        return value
    
    def _add_analysis_layer(self, all_data: List[Dict[str, Any]], result: XBRLAnalysisResult) -> Dict[str, Any]:
        """Add analytical insights and compute trends"""
        analysis = {
            "shareholding_breakdown": {},
            "year_over_year_trends": {},
            "key_insights": [],
            "computed_percentages": {},
            "major_changes": [],
            "risk_opportunities": []
        }
        
        # Aggregate shareholding data by period
        periods_data = {}
        
        for data in all_data:
            period = data.get("metadata", {}).get("period", "Unknown")
            if period not in periods_data:
                periods_data[period] = {
                    "promoter": 0, "fpi": 0, "institutional": 0, 
                    "retail": 0, "foreign": 0, "total_shareholders": 0
                }
            
            # Extract and categorize shareholding data
            if "shareholding_patterns" in data:
                for pattern in data["shareholding_patterns"]:
                    category = pattern["category"].lower()
                    value = pattern["value"]
                    
                    # Extract numeric values
                    numbers = re.findall(r'([0-9.]+)', str(value))
                    numeric_value = float(numbers[0]) if numbers else 0
                    
                    # Categorize holdings
                    if 'promot' in category:
                        periods_data[period]["promoter"] += numeric_value
                    elif any(term in category for term in ['foreign', 'fpi']):
                        periods_data[period]["foreign"] += numeric_value
                        if 'fpi' in category:
                            periods_data[period]["fpi"] += numeric_value
                    elif any(term in category for term in ['institution', 'mutual', 'insurance']):
                        periods_data[period]["institutional"] += numeric_value
                    elif 'retail' in category or 'public' in category:
                        periods_data[period]["retail"] += numeric_value
                    elif 'shareholder' in category and 'number' in category:
                        periods_data[period]["total_shareholders"] = numeric_value
        
        # Compute year-over-year trends
        sorted_periods = sorted(periods_data.keys())
        if len(sorted_periods) > 1:
            current = periods_data[sorted_periods[-1]]
            previous = periods_data[sorted_periods[-2]]
            
            for category in ["promoter", "foreign", "institutional", "retail"]:
                current_val = current.get(category, 0)
                previous_val = previous.get(category, 0)
                change = current_val - previous_val
                
                if previous_val > 0:
                    change_pct = (change / previous_val) * 100
                    analysis["year_over_year_trends"][category] = {
                        "change": change,
                        "change_percentage": change_pct,
                        "current": current_val,
                        "previous": previous_val
                    }
        
        # Generate insights
        analysis["key_insights"] = self._generate_narrative_insights(periods_data, analysis["year_over_year_trends"])
        analysis["shareholding_breakdown"] = periods_data
        
        return analysis
    
    def _generate_narrative_insights(self, periods_data: Dict, trends: Dict) -> List[str]:
        """Generate narrative insights from the data"""
        insights = []
        
        if not periods_data:
            return ["No sufficient data available for trend analysis"]
        
        # Get latest period data
        latest_period = max(periods_data.keys()) if periods_data else None
        if not latest_period:
            return insights
        
        latest = periods_data[latest_period]
        
        # Foreign investment insights
        if trends.get("foreign", {}).get("change_percentage"):
            change_pct = trends["foreign"]["change_percentage"]
            if change_pct > 0:
                insights.append(f"🔥 Foreign participation increased by {change_pct:.1f}% YoY, signaling growing international confidence")
            else:
                insights.append(f"⚠️ Foreign participation declined by {abs(change_pct):.1f}% YoY, worth monitoring for market sentiment")
        
        # Promoter insights
        if trends.get("promoter", {}).get("change_percentage"):
            change_pct = trends["promoter"]["change_percentage"]
            if change_pct < -2:
                insights.append(f"📉 Promoter holdings decreased by {abs(change_pct):.1f}%, indicating potential dilution or strategic changes")
            elif change_pct > 2:
                insights.append(f"📈 Promoter holdings increased by {change_pct:.1f}%, showing strong management confidence")
        
        # Institutional insights
        if trends.get("institutional", {}).get("change_percentage"):
            change_pct = trends["institutional"]["change_percentage"]
            if change_pct > 5:
                insights.append(f"🏦 Institutional participation surged by {change_pct:.1f}%, reflecting strong institutional interest")
        
        # Shareholder concentration insights
        total_shareholders = latest.get("total_shareholders", 0)
        if total_shareholders > 0:
            if total_shareholders > 100000:
                insights.append(f"🌐 Broad retail participation with {int(total_shareholders):,} shareholders indicates strong public interest")
            elif total_shareholders < 1000:
                insights.append(f"⚠️ Limited shareholder base ({int(total_shareholders):,}) suggests concentrated ownership")
        
        return insights
