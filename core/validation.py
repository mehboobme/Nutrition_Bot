"""Input validation utilities for the RAG application.

Provides comprehensive validation for user inputs, queries, and configuration.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result of a validation check."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ValidationError:
    """Details of a validation error."""
    field: str
    message: str
    code: str
    severity: ValidationResult = ValidationResult.INVALID


class InputValidator:
    """Validator for user inputs and queries.
    
    Performs validation on:
    - Query length and content
    - User IDs
    - Injection attacks
    - Encoding issues
    """
    
    # Constraints
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 5000
    MAX_USER_ID_LENGTH = 100
    
    # Patterns
    INJECTION_PATTERNS = [
        r"(?i)\b(drop|delete|truncate|update|insert)\s+(table|from|into)",
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
    ]
    
    # Compile patterns for efficiency
    _injection_regex = [re.compile(p) for p in INJECTION_PATTERNS]
    
    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a user query.
        
        Args:
            query: The query string to validate.
            
        Returns:
            Tuple of (is_valid, list of errors).
        """
        errors = []
        
        # Check for None/empty
        if not query:
            errors.append(ValidationError(
                field="query",
                message="Query cannot be empty",
                code="EMPTY_QUERY"
            ))
            return False, errors
        
        # Strip and normalize
        query = query.strip()
        
        # Check length
        if len(query) < cls.MIN_QUERY_LENGTH:
            errors.append(ValidationError(
                field="query",
                message=f"Query must be at least {cls.MIN_QUERY_LENGTH} characters",
                code="QUERY_TOO_SHORT"
            ))
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            errors.append(ValidationError(
                field="query",
                message=f"Query cannot exceed {cls.MAX_QUERY_LENGTH} characters",
                code="QUERY_TOO_LONG"
            ))
        
        # Check for injection attempts
        for pattern in cls._injection_regex:
            if pattern.search(query):
                errors.append(ValidationError(
                    field="query",
                    message="Query contains potentially harmful content",
                    code="INJECTION_DETECTED"
                ))
                break
        
        # Check for valid UTF-8
        try:
            query.encode('utf-8').decode('utf-8')
        except UnicodeError:
            errors.append(ValidationError(
                field="query",
                message="Query contains invalid characters",
                code="INVALID_ENCODING"
            ))
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_user_id(cls, user_id: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a user ID.
        
        Args:
            user_id: The user ID to validate.
            
        Returns:
            Tuple of (is_valid, list of errors).
        """
        errors = []
        
        if not user_id:
            errors.append(ValidationError(
                field="user_id",
                message="User ID cannot be empty",
                code="EMPTY_USER_ID"
            ))
            return False, errors
        
        user_id = user_id.strip()
        
        if len(user_id) > cls.MAX_USER_ID_LENGTH:
            errors.append(ValidationError(
                field="user_id",
                message=f"User ID cannot exceed {cls.MAX_USER_ID_LENGTH} characters",
                code="USER_ID_TOO_LONG"
            ))
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[\w\-\.@]+$', user_id):
            errors.append(ValidationError(
                field="user_id",
                message="User ID contains invalid characters",
                code="INVALID_USER_ID_CHARS"
            ))
        
        return len(errors) == 0, errors
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """
        Sanitize a query by removing potentially harmful content.
        
        Args:
            query: The query to sanitize.
            
        Returns:
            Sanitized query string.
        """
        if not query:
            return ""
        
        # Strip whitespace
        query = query.strip()
        
        # Remove HTML tags
        query = re.sub(r'<[^>]+>', '', query)
        
        # Remove script-like content
        query = re.sub(r'javascript:', '', query, flags=re.IGNORECASE)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    @classmethod
    def sanitize_for_logging(cls, text: str, max_length: int = 200) -> str:
        """
        Sanitize text for safe logging.
        
        Args:
            text: Text to sanitize.
            max_length: Maximum length to include.
            
        Returns:
            Sanitized text safe for logging.
        """
        if not text:
            return ""
        
        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Remove newlines for log readability
        text = text.replace('\n', ' ').replace('\r', '')
        
        return text


def validate_input(func):
    """
    Decorator to validate query input before processing.
    
    Automatically validates the 'query' parameter if present.
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Look for query in kwargs or positional args
        query = kwargs.get('query')
        if query is None and len(args) > 1:
            # Assuming query is the second positional arg (after self)
            query = args[1] if len(args) > 1 else None
        
        if query is not None:
            is_valid, errors = InputValidator.validate_query(query)
            if not is_valid:
                error_messages = [f"{e.field}: {e.message}" for e in errors]
                raise ValueError(f"Invalid input: {'; '.join(error_messages)}")
        
        return func(*args, **kwargs)
    
    return wrapper
