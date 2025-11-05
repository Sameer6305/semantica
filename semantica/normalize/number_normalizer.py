"""
Number and Quantity Normalization Module

Handles normalization of numbers, quantities, and numerical expressions.

Key Features:
    - Number format standardization
    - Unit conversion and normalization
    - Currency handling
    - Percentage processing
    - Scientific notation handling

Main Classes:
    - NumberNormalizer: Main number normalization class
    - UnitConverter: Unit conversion engine
    - CurrencyNormalizer: Currency processing
    - ScientificNotationHandler: Scientific notation processor
"""

import re
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


class NumberNormalizer:
    """
    Number and quantity normalization handler.
    
    • Normalizes numbers to standard formats
    • Handles various number representations
    • Processes quantities and units
    • Manages currency and percentage values
    • Standardizes numerical expressions
    • Supports multiple number systems
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize number normalizer.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("number_normalizer")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.unit_converter = UnitConverter(**self.config)
        self.currency_normalizer = CurrencyNormalizer(**self.config)
        self.scientific_handler = ScientificNotationHandler(**self.config)
    
    def normalize_number(self, number_input: Union[str, int, float], **options) -> Union[int, float]:
        """
        Normalize number to standard format.
        
        Args:
            number_input: Number input (string, int, or float)
            **options: Normalization options
        
        Returns:
            Normalized number
        """
        if isinstance(number_input, (int, float)):
            return number_input
        
        if not isinstance(number_input, str):
            raise ValidationError(f"Unsupported number input type: {type(number_input)}")
        
        # Remove formatting characters
        cleaned = number_input.replace(',', '').replace(' ', '').strip()
        
        # Handle percentages
        if '%' in cleaned:
            cleaned = cleaned.replace('%', '')
            value = float(cleaned) / 100
            return value
        
        # Handle scientific notation
        if 'e' in cleaned.lower() or 'E' in cleaned:
            parsed = self.scientific_handler.parse_scientific_notation(cleaned)
            return parsed
        
        # Parse as float or int
        try:
            if '.' in cleaned:
                return float(cleaned)
            else:
                return int(cleaned)
        except ValueError:
            raise ValidationError(f"Unable to parse number: {number_input}")
    
    def normalize_quantity(self, quantity_input: str, **options) -> Dict[str, Any]:
        """
        Normalize quantity with units.
        
        Args:
            quantity_input: Quantity string (e.g., "5 kg", "10 meters")
            **options: Normalization options
        
        Returns:
            Normalized quantity dictionary
        """
        # Parse quantity and unit
        pattern = r'([\d.,\s]+)\s*([a-zA-Z]+)'
        match = re.search(pattern, quantity_input)
        
        if match:
            value_str = match.group(1).replace(',', '').replace(' ', '')
            unit = match.group(2).lower()
            
            try:
                value = float(value_str)
                
                # Normalize unit
                normalized_unit = self.unit_converter.normalize_unit(unit)
                
                return {
                    "value": value,
                    "unit": normalized_unit,
                    "original": quantity_input
                }
            except ValueError:
                raise ValidationError(f"Unable to parse quantity: {quantity_input}")
        else:
            raise ValidationError(f"Invalid quantity format: {quantity_input}")
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert value between units.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
        
        Returns:
            Converted value
        """
        return self.unit_converter.convert_units(value, from_unit, to_unit)
    
    def process_currency(self, currency_input: str, **options) -> Dict[str, Any]:
        """
        Process currency values.
        
        Args:
            currency_input: Currency string (e.g., "$100", "100 USD")
            **options: Processing options
        
        Returns:
            Normalized currency dictionary
        """
        return self.currency_normalizer.normalize_currency(currency_input, **options)


class UnitConverter:
    """
    Unit conversion engine.
    
    • Converts between different units
    • Handles various unit systems
    • Manages conversion factors
    • Processes compound units
    • Handles unit validation
    """
    
    def __init__(self, **config):
        """
        Initialize unit converter.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("unit_converter")
        self.config = config
        
        # Unit conversion factors (to base unit)
        self.conversion_factors = {
            # Length
            "meter": 1.0, "meters": 1.0, "m": 1.0,
            "kilometer": 1000.0, "kilometers": 1000.0, "km": 1000.0,
            "centimeter": 0.01, "centimeters": 0.01, "cm": 0.01,
            "millimeter": 0.001, "millimeters": 0.001, "mm": 0.001,
            "inch": 0.0254, "inches": 0.0254, "in": 0.0254,
            "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
            "yard": 0.9144, "yards": 0.9144, "yd": 0.9144,
            "mile": 1609.34, "miles": 1609.34, "mi": 1609.34,
            
            # Weight
            "kilogram": 1.0, "kilograms": 1.0, "kg": 1.0,
            "gram": 0.001, "grams": 0.001, "g": 0.001,
            "pound": 0.453592, "pounds": 0.453592, "lb": 0.453592,
            "ounce": 0.0283495, "ounces": 0.0283495, "oz": 0.0283495,
            
            # Volume
            "liter": 1.0, "liters": 1.0, "l": 1.0,
            "milliliter": 0.001, "milliliters": 0.001, "ml": 0.001,
            "gallon": 3.78541, "gallons": 3.78541, "gal": 3.78541,
        }
        
        # Unit categories
        self.unit_categories = {
            "length": ["meter", "kilometer", "centimeter", "millimeter", "inch", "foot", "yard", "mile"],
            "weight": ["kilogram", "gram", "pound", "ounce"],
            "volume": ["liter", "milliliter", "gallon"],
        }
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert value between units.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
        
        Returns:
            Converted value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Validate units
        if not self.validate_units(from_unit, to_unit):
            raise ValidationError(f"Cannot convert from {from_unit} to {to_unit}")
        
        # Get conversion factors
        from_factor = self.get_conversion_factor(from_unit, "base")
        to_factor = self.get_conversion_factor(to_unit, "base")
        
        # Convert to base unit, then to target unit
        base_value = value * from_factor
        converted_value = base_value / to_factor
        
        return converted_value
    
    def validate_units(self, from_unit: str, to_unit: str) -> bool:
        """
        Validate unit conversion compatibility.
        
        Args:
            from_unit: Source unit
            to_unit: Target unit
        
        Returns:
            True if units are compatible
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Check if both units exist
        if from_unit not in self.conversion_factors or to_unit not in self.conversion_factors:
            return False
        
        # Check if units are in same category
        from_category = None
        to_category = None
        
        for category, units in self.unit_categories.items():
            if from_unit in units:
                from_category = category
            if to_unit in units:
                to_category = category
        
        return from_category == to_category
    
    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """
        Get conversion factor between units.
        
        Args:
            from_unit: Source unit
            to_unit: Target unit
        
        Returns:
            Conversion factor
        """
        from_unit = from_unit.lower()
        
        if to_unit == "base":
            return self.conversion_factors.get(from_unit, 1.0)
        
        to_unit = to_unit.lower()
        from_factor = self.conversion_factors.get(from_unit, 1.0)
        to_factor = self.conversion_factors.get(to_unit, 1.0)
        
        return from_factor / to_factor
    
    def normalize_unit(self, unit: str) -> str:
        """
        Normalize unit name to standard form.
        
        Args:
            unit: Unit name
        
        Returns:
            Normalized unit name
        """
        unit_lower = unit.lower()
        
        # Map to standard unit
        unit_map = {
            "m": "meter",
            "km": "kilometer",
            "cm": "centimeter",
            "mm": "millimeter",
            "kg": "kilogram",
            "g": "gram",
            "lb": "pound",
            "oz": "ounce",
            "l": "liter",
            "ml": "milliliter",
        }
        
        return unit_map.get(unit_lower, unit_lower)


class CurrencyNormalizer:
    """
    Currency normalization engine.
    
    • Normalizes currency values and codes
    • Handles currency conversion
    • Manages exchange rates
    • Processes currency symbols
    • Handles multiple currencies
    """
    
    def __init__(self, **config):
        """
        Initialize currency normalizer.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("currency_normalizer")
        self.config = config
        
        # Currency symbols and codes
        self.currency_symbols = {
            "$": "USD",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            "₹": "INR",
            "₽": "RUB",
            "₩": "KRW",
            "₪": "ILS",
            "₦": "NGN",
            "₨": "PKR",
        }
        
        self.currency_codes = ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD", "CHF", "SEK", "NOK", "DKK"]
    
    def normalize_currency(self, currency_input: str, **options) -> Dict[str, Any]:
        """
        Normalize currency value and code.
        
        Args:
            currency_input: Currency string
            **options: Normalization options
        
        Returns:
            Normalized currency dictionary
        """
        # Extract currency symbol or code
        currency_code = None
        amount = None
        
        # Check for currency symbol
        for symbol, code in self.currency_symbols.items():
            if symbol in currency_input:
                currency_code = code
                # Remove symbol and extract amount
                amount_str = currency_input.replace(symbol, "").strip()
                amount_str = amount_str.replace(',', '').replace(' ', '')
                try:
                    amount = float(amount_str)
                except ValueError:
                    pass
                break
        
        # Check for currency code
        if not currency_code:
            for code in self.currency_codes:
                if code in currency_input.upper():
                    currency_code = code
                    amount_str = currency_input.replace(code, "").replace(code.lower(), "").strip()
                    amount_str = amount_str.replace(',', '').replace(' ', '')
                    try:
                        amount = float(amount_str)
                    except ValueError:
                        pass
                    break
        
        # Extract amount if not found
        if amount is None:
            amount_str = re.sub(r'[^\d.,]', '', currency_input)
            amount_str = amount_str.replace(',', '').replace(' ', '')
            try:
                amount = float(amount_str)
            except ValueError:
                amount = None
        
        # Default to USD if no currency found
        if not currency_code:
            currency_code = options.get("default_currency", "USD")
        
        return {
            "amount": amount,
            "currency": currency_code,
            "original": currency_input
        }
    
    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert currency between different currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
        
        Returns:
            Converted amount
        """
        # Note: This is a placeholder. In production, you'd fetch exchange rates
        self.logger.warning("Currency conversion requires exchange rate API")
        return amount
    
    def validate_currency_code(self, currency_code: str) -> bool:
        """
        Validate currency code.
        
        Args:
            currency_code: Currency code to validate
        
        Returns:
            True if valid
        """
        return currency_code.upper() in self.currency_codes


class ScientificNotationHandler:
    """
    Scientific notation processing engine.
    
    • Handles scientific notation numbers
    • Processes exponential formats
    • Manages precision and significant digits
    • Converts between formats
    """
    
    def __init__(self, **config):
        """
        Initialize scientific notation handler.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("scientific_notation_handler")
        self.config = config
    
    def parse_scientific_notation(self, number_string: str) -> float:
        """
        Parse scientific notation number.
        
        Args:
            number_string: Scientific notation string
        
        Returns:
            Parsed number as float
        """
        try:
            return float(number_string)
        except ValueError:
            raise ValidationError(f"Invalid scientific notation: {number_string}")
    
    def convert_to_scientific(self, number: float, precision: Optional[int] = None) -> str:
        """
        Convert number to scientific notation.
        
        Args:
            number: Number to convert
            precision: Precision (number of decimal places)
        
        Returns:
            Scientific notation string
        """
        if precision is not None:
            return f"{number:.{precision}e}"
        else:
            return f"{number:e}"
    
    def normalize_precision(self, number: float, significant_digits: int) -> float:
        """
        Normalize number precision.
        
        Args:
            number: Number to normalize
            significant_digits: Number of significant digits
        
        Returns:
            Normalized number
        """
        if number == 0:
            return 0.0
        
        # Calculate order of magnitude
        magnitude = abs(number)
        order = 0
        while magnitude >= 10:
            magnitude /= 10
            order += 1
        while magnitude < 1:
            magnitude *= 10
            order -= 1
        
        # Round to significant digits
        rounded = round(magnitude, significant_digits - 1)
        
        return rounded * (10 ** order)
