"""
XBRL Analysis Tools for processing financial data from XBRL files.
"""
import os
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from pydantic import BaseModel, Field

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
        self.temp_cleanup_age = timedelta(minutes=30)
        self.entity_mapping = {}
    
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
                    logger.error(f"Error processing file {file_path}: {e}")
            
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
            
            # Try to parse as XML first
            try:
                root = ET.fromstring(content)
                return self._extract_xml_data(root, analysis_type)
            except ET.ParseError:
                # If XML parsing fails, try text-based extraction
                logger.warning(f"XML parsing failed for {file_path}, trying text-based extraction")
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
            "entity_mapping": {}
        }
        
        # Extract entity mappings first
        data["entity_mapping"] = self._extract_entity_mappings(root)
        
        # Extract basic metadata
        data["metadata"]["filing_date"] = self._find_element_text(root, ["FilingDate", "ReportingDate", "PeriodEndDate"])
        data["metadata"]["company_name"] = self._find_element_text(root, ["CompanyName", "EntityName", "NameOfCompany"])
        data["metadata"]["period"] = self._find_element_text(root, ["Period", "ReportingPeriod", "Quarter"])
        
        if analysis_type == "shareholding":
            data = self._extract_shareholding_xml_data(root, data)
        elif analysis_type == "governance":
            data = self._extract_governance_xml_data(root, data)
        
        return data
    
    def _extract_shareholding_xml_data(self, root: ET.Element, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shareholding-specific data from XML"""
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
        return data
    
    def _extract_governance_xml_data(self, root: ET.Element, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract governance-specific data from XML"""
        governance_items = []
        
        for elem in root.iter():
            tag_lower = elem.tag.lower() if elem.tag else ""
            text = elem.text or ""
            
            if any(keyword in tag_lower for keyword in ['board', 'director', 'committee', 'audit', 'governance']):
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
            patterns = {
                "promoter_percentage": r"promoter[s]?\s*[:=]\s*([0-9.]+)%?",
                "public_percentage": r"public\s*[:=]\s*([0-9.]+)%?",
                "foreign_percentage": r"foreign\s*[:=]\s*([0-9.]+)%?",
                "institutional_percentage": r"institutional\s*[:=]\s*([0-9.]+)%?"
            }
        elif analysis_type == "governance":
            patterns = {
                "board_size": r"board\s+size\s*[:=]\s*([0-9]+)",
                "independent_directors": r"independent\s+director[s]?\s*[:=]\s*([0-9]+)",
                "audit_committee": r"audit\s+committee\s+member[s]?\s*[:=]\s*([0-9]+)",
                "women_directors": r"women\s+director[s]?\s*[:=]\s*([0-9]+)"
            }
        else:
            patterns = {}
        
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
                    category = pattern.get("category", "Unknown")
                    value = pattern.get("value", "")
                    
                    # Categorize patterns
                    if any(term in category.lower() for term in ['foreign', 'fpi', 'overseas']):
                        foreign_shareholders.append(pattern)
                    elif any(term in category.lower() for term in ['sharehold', 'equity', 'stake']):
                        top_shareholders.append(pattern)
                    
                    # Store in summary
                    if category not in shareholding_summary:
                        shareholding_summary[category] = []
                    shareholding_summary[category].append(value)
            
            if "extracted_values" in data:
                shareholding_summary.update(data["extracted_values"])
        
        # Add analysis layer
        analysis_insights = self._add_analysis_layer(all_data, result)
        
        # Store data in result
        result.shareholding_data = shareholding_summary
        result.shareholding_data["foreign_shareholding_details"] = foreign_shareholders
        result.shareholding_data["major_shareholders"] = top_shareholders
        result.shareholding_data["analysis_insights"] = analysis_insights
        
        # Generate insights
        key_insights = analysis_insights.get("key_insights", [])
        if not key_insights and shareholding_summary:
            key_insights = [
                f"Extracted shareholding data from {len(all_data)} XBRL files",
                f"Identified {len(shareholding_summary)} distinct shareholding categories",
                f"Found {len(foreign_shareholders)} foreign investment entries",
                f"Identified {len(top_shareholders)} major stakeholder entries"
            ]
        
        result.insights = key_insights
        result.summary = f"Analyzed {len(all_data)} XBRL files for shareholding patterns. Generated {len(key_insights)} key insights."
        
        return result
    
    def _analyze_governance_data(self, all_data: List[Dict[str, Any]], result: XBRLAnalysisResult) -> XBRLAnalysisResult:
        """Analyze corporate governance data and generate insights"""
        governance_summary = {}
        key_insights = []
        
        # Aggregate data from all files
        for data in all_data:
            if "governance_items" in data:
                for item in data["governance_items"]:
                    category = item.get("category", "Unknown")
                    value = item.get("value", "")
                    
                    if category not in governance_summary:
                        governance_summary[category] = []
                    governance_summary[category].append(value)
            
            if "extracted_values" in data:
                governance_summary.update(data["extracted_values"])
        
        result.governance_data = governance_summary
        
        # Generate insights
        if governance_summary:
            key_insights.append("Corporate governance data successfully extracted from XBRL files")
            
            # Look for board composition
            board_keys = [k for k in governance_summary.keys() if any(term in k.lower() for term in ['board', 'director'])]
            if board_keys:
                key_insights.append(f"Board composition information found across {len(board_keys)} categories")
            
            # Look for committee information
            committee_keys = [k for k in governance_summary.keys() if 'committee' in k.lower()]
            if committee_keys:
                key_insights.append(f"Committee structure data available for {len(committee_keys)} committees")
            
            # Generate key metrics
            for key, values in governance_summary.items():
                if isinstance(values, list) and values:
                    result.key_metrics.append({
                        "metric": key,
                        "value": values[0] if len(values) == 1 else values,
                        "category": "governance"
                    })
        
        result.insights = key_insights
        result.summary = f"Analyzed {len(all_data)} XBRL files for corporate governance. Extracted {len(governance_summary)} data points."
        
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
                        if datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff_time:
                            os.remove(file_path)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count
    
    def _extract_entity_mappings(self, root: ET.Element) -> Dict[str, str]:
        """Extract context ID to entity name mappings from XBRL"""
        mappings = {}
        
        # Look for context definitions
        for elem in root.iter():
            if elem.tag and ('context' in elem.tag.lower() or 'entity' in elem.tag.lower()):
                context_id = elem.get('id', '')
                entity_name = elem.text or ''
                
                if context_id and entity_name:
                    mappings[context_id] = entity_name
        
        return mappings
    
    def _resolve_context_reference(self, value: str, entity_mapping: Dict[str, str]) -> str:
        """Resolve context references to actual entity names"""
        if not value or not isinstance(value, str):
            return value
        
        # Check if value looks like a context reference
        context_pattern = re.match(r'^(context|c_?)(\d+)$', value.lower().strip())
        if context_pattern:
            context_key = context_pattern.group(0)
            return entity_mapping.get(context_key, value)
        
        # Check if value contains context references mixed with other text
        context_refs = re.findall(r'\b(context\d+|c_\d+)\b', value.lower())
        if context_refs:
            resolved_value = value
            for ref in context_refs:
                entity_name = entity_mapping.get(ref, ref)
                resolved_value = resolved_value.replace(ref, entity_name)
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
                    "promoter": 0,
                    "foreign": 0,
                    "institutional": 0,
                    "retail": 0,
                    "total_shareholders": 0
                }
            
            # Extract percentage data from shareholding patterns
            if "shareholding_patterns" in data:
                for pattern in data["shareholding_patterns"]:
                    category = pattern.get("category", "").lower()
                    value_str = pattern.get("value", "")
                    
                    # Try to extract numerical values
                    percentage_match = re.search(r'(\d+\.?\d*)%?', str(value_str))
                    if percentage_match:
                        percentage = float(percentage_match.group(1))
                        
                        if any(term in category for term in ['promot', 'management']):
                            periods_data[period]["promoter"] = max(periods_data[period]["promoter"], percentage)
                        elif any(term in category for term in ['foreign', 'fpi']):
                            periods_data[period]["foreign"] = max(periods_data[period]["foreign"], percentage)
                        elif any(term in category for term in ['institution', 'mutual']):
                            periods_data[period]["institutional"] = max(periods_data[period]["institutional"], percentage)
                        elif any(term in category for term in ['public', 'retail']):
                            periods_data[period]["retail"] = max(periods_data[period]["retail"], percentage)
        
        # Compute year-over-year trends
        sorted_periods = sorted(periods_data.keys())
        if len(sorted_periods) > 1:
            current = periods_data[sorted_periods[-1]]
            previous = periods_data[sorted_periods[-2]]
            
            for category in ["promoter", "foreign", "institutional", "retail"]:
                if previous[category] > 0:
                    change = ((current[category] - previous[category]) / previous[category]) * 100
                    analysis["year_over_year_trends"][category] = {
                        "current": current[category],
                        "previous": previous[category],
                        "change_percentage": change
                    }
        
        # Generate insights
        analysis["key_insights"] = self._generate_narrative_insights(periods_data, analysis["year_over_year_trends"])
        analysis["shareholding_breakdown"] = periods_data
        
        return analysis
    
    def _generate_narrative_insights(self, periods_data: Dict, trends: Dict) -> List[str]:
        """Generate narrative insights from the data"""
        insights = []
        
        if not periods_data:
            insights.append("Shareholding pattern data extracted from XBRL filings")
            return insights
        
        # Get latest period data
        latest_period = max(periods_data.keys()) if periods_data else None
        if not latest_period:
            insights.append("Basic shareholding information available from regulatory filings")
            return insights
        
        latest = periods_data[latest_period]
        
        # Trend-based insights
        for category, trend_info in trends.items():
            change_pct = trend_info.get("change_percentage", 0)
            current = trend_info.get("current", 0)
            
            if abs(change_pct) > 1:  # Only report significant changes
                if change_pct > 0:
                    insights.append(f"{category.title()} holdings increased by {change_pct:.1f}% to {current:.1f}%")
                else:
                    insights.append(f"{category.title()} holdings decreased by {abs(change_pct):.1f}% to {current:.1f}%")
        
        # Composition insights
        total_holdings = sum([latest.get(k, 0) for k in ['promoter', 'foreign', 'institutional', 'retail']])
        if total_holdings > 50:  # Meaningful data threshold
            insights.append(f"Comprehensive shareholding structure identified with {total_holdings:.1f}% coverage")
        
        # Default insight if no specific patterns found
        if not insights:
            insights.append("Shareholding pattern analysis completed from BSE XBRL regulatory filings")
        
        return insights
