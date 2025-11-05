"""
Repository Ingestion Module

Handles Git repository processing and code analysis.

Key Features:
    - Git repository cloning and analysis
    - Code file extraction
    - Commit history processing
    - Branch and tag analysis
    - Code structure analysis

Main Classes:
    - RepoIngestor: Main repository ingestion class
    - GitAnalyzer: Git repository analyzer
    - CodeExtractor: Code content extractor
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import git

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


@dataclass
class CodeFile:
    """Code file representation."""
    
    path: str
    name: str
    language: str
    content: str
    size: int
    lines: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommitInfo:
    """Commit information representation."""
    
    hash: str
    message: str
    author: str
    date: datetime
    files_changed: List[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


class CodeExtractor:
    """
    Code content extraction and processing.
    
    Extracts code content from various file types
    and processes different programming languages.
    """
    
    # File extension to language mapping
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'matlab',
        '.sh': 'shell',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.txt': 'text',
    }
    
    def __init__(self, **config):
        """
        Initialize code extractor.
        
        Args:
            **config: Extraction configuration
        """
        self.logger = get_logger("code_extractor")
        self.config = config
    
    def extract_file_content(self, file_path: Path, language: Optional[str] = None) -> CodeFile:
        """
        Extract content from code file.
        
        Args:
            file_path: Path to code file
            language: Programming language (auto-detected if not provided)
            
        Returns:
            CodeFile: Extracted code file data
        """
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Detect language if not provided
        if language is None:
            language = self._detect_language(file_path)
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            self.logger.warning(f"Failed to read file {file_path}: {e}")
            content = ""
        
        # Get file stats
        stat = file_path.stat()
        lines = content.count('\n') + (1 if content else 0)
        
        # Extract metadata
        metadata = {
            "extension": file_path.suffix,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime)
        }
        
        # Extract structure if supported
        if language in ['python', 'javascript', 'typescript', 'java']:
            metadata["structure"] = self._extract_structure(content, language)
        
        return CodeFile(
            path=str(file_path),
            name=file_path.name,
            language=language,
            content=content,
            size=stat.st_size,
            lines=lines,
            metadata=metadata
        )
    
    def extract_documentation(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract documentation from code file.
        
        Args:
            file_path: Path to code file
            
        Returns:
            dict: Documentation content
        """
        language = self._detect_language(file_path)
        docs = {
            "docstrings": [],
            "comments": [],
            "readme": None
        }
        
        if file_path.name.lower() in ['readme.md', 'readme.txt', 'readme']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                docs["readme"] = f.read()
            return docs
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return docs
        
        # Extract docstrings (Python style)
        if language == 'python':
            docstring_pattern = r'"""(.*?)"""'
            matches = re.findall(docstring_pattern, content, re.DOTALL)
            docs["docstrings"] = [m.strip() for m in matches]
        
        # Extract comments
        comment_patterns = {
            'python': r'#\s*(.*)',
            'javascript': r'//\s*(.*)',
            'java': r'//\s*(.*)',
            'c': r'//\s*(.*)',
            'cpp': r'//\s*(.*)',
        }
        
        if language in comment_patterns:
            pattern = comment_patterns[language]
            matches = re.findall(pattern, content, re.MULTILINE)
            docs["comments"] = [m.strip() for m in matches if m.strip()]
        
        return docs
    
    def analyze_dependencies(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze file dependencies and imports.
        
        Args:
            file_path: Path to code file
            
        Returns:
            dict: Dependency analysis
        """
        language = self._detect_language(file_path)
        dependencies = {
            "external": [],
            "internal": [],
            "language": language
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return dependencies
        
        # Python imports
        if language == 'python':
            import_pattern = r'^(?:from|import)\s+([\w.]+)'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            dependencies["external"] = matches
        
        # JavaScript/TypeScript imports
        elif language in ['javascript', 'typescript']:
            import_pattern = r"(?:import|require)\(?['\"]([^'\"]+)['\"]"
            matches = re.findall(import_pattern, content)
            dependencies["external"] = matches
        
        # Java imports
        elif language == 'java':
            import_pattern = r'^import\s+([\w.]+)'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            dependencies["external"] = matches
        
        return dependencies
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(ext, 'unknown')
    
    def _extract_structure(self, content: str, language: str) -> Dict[str, Any]:
        """Extract code structure (functions, classes, etc.)."""
        structure = {
            "classes": [],
            "functions": [],
            "imports": []
        }
        
        if language == 'python':
            # Extract classes
            class_pattern = r'^class\s+(\w+)'
            structure["classes"] = re.findall(class_pattern, content, re.MULTILINE)
            
            # Extract functions
            func_pattern = r'^def\s+(\w+)'
            structure["functions"] = re.findall(func_pattern, content, re.MULTILINE)
        
        elif language in ['javascript', 'typescript']:
            # Extract classes
            class_pattern = r'class\s+(\w+)'
            structure["classes"] = re.findall(class_pattern, content)
            
            # Extract functions
            func_pattern = r'function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\(|(\w+)\s*:\s*(?:async\s+)?\(.*?\)\s*=>
            matches = re.findall(func_pattern, content)
            structure["functions"] = [m[0] or m[1] or m[2] for m in matches if any(m)]
        
        return structure


class GitAnalyzer:
    """
    Git repository analysis and statistics.
    
    Analyzes repository structure and history,
    calculates code metrics and statistics.
    """
    
    def __init__(self, **config):
        """
        Initialize Git analyzer.
        
        Args:
            **config: Analyzer configuration
        """
        self.logger = get_logger("git_analyzer")
        self.config = config
    
    def analyze_structure(self, repo_path: Path) -> Dict[str, Any]:
        """
        Analyze repository file structure.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            dict: Structure analysis results
        """
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "directory_structure": {},
            "max_depth": 0
        }
        
        def analyze_dir(path: Path, depth: int = 0):
            structure["max_depth"] = max(structure["max_depth"], depth)
            
            try:
                items = list(path.iterdir())
                for item in items:
                    if item.is_file():
                        structure["total_files"] += 1
                        ext = item.suffix.lower() or 'no_extension'
                        structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
                    elif item.is_dir() and item.name not in ['.git', '__pycache__', 'node_modules']:
                        structure["total_directories"] += 1
                        analyze_dir(item, depth + 1)
            except PermissionError:
                pass
        
        analyze_dir(repo_path)
        return structure
    
    def calculate_metrics(self, repo_path: Path) -> Dict[str, Any]:
        """
        Calculate repository metrics and statistics.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            dict: Metrics dictionary
        """
        metrics = {
            "total_lines": 0,
            "total_files": 0,
            "lines_by_language": {},
            "files_by_language": {}
        }
        
        extractor = CodeExtractor()
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and file_path.name not in ['.gitignore', '.gitkeep']:
                try:
                    code_file = extractor.extract_file_content(file_path)
                    metrics["total_files"] += 1
                    metrics["total_lines"] += code_file.lines
                    
                    lang = code_file.language
                    metrics["lines_by_language"][lang] = metrics["lines_by_language"].get(lang, 0) + code_file.lines
                    metrics["files_by_language"][lang] = metrics["files_by_language"].get(lang, 0) + 1
                except Exception:
                    continue
        
        return metrics
    
    def detect_patterns(self, repo_path: Path) -> Dict[str, Any]:
        """
        Detect code patterns and conventions.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            dict: Pattern analysis results
        """
        patterns = {
            "naming_conventions": {},
            "file_organization": {},
            "common_structures": []
        }
        
        # Analyze naming patterns
        python_files = list(repo_path.rglob('*.py'))
        if python_files:
            patterns["naming_conventions"]["python"] = "detected"
        
        return patterns


class RepoIngestor:
    """
    Git repository ingestion handler.
    
    Clones and analyzes Git repositories,
    extracts code files and documentation,
    and processes commit history and metadata.
    
    Attributes:
        git_client: Git client for repository operations
        code_extractor: Code content extraction engine
        analyzer: Repository analysis tools
        supported_platforms: List of supported Git platforms
        
    Methods:
        ingest_repository(): Clone and process repository
        analyze_commits(): Analyze commit history
        extract_code_files(): Extract and process code files
        get_repository_info(): Get repository metadata
    """
    
    SUPPORTED_PLATFORMS = ['github', 'gitlab', 'bitbucket', 'generic']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize repository ingestor.
        
        Args:
            config: Repository ingestion configuration
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("repo_ingestor")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize code extractor
        self.code_extractor = CodeExtractor(**self.config)
        
        # Initialize analyzer
        self.analyzer = GitAnalyzer(**self.config)
        
        # Temporary directory for cloning
        self.temp_dir = None
    
    def ingest_repository(self, repo_url: str, **options) -> Dict[str, Any]:
        """
        Ingest and process a Git repository.
        
        Args:
            repo_url: Repository URL
            **options: Processing options:
                - branch: Specific branch to checkout
                - depth: Clone depth (for shallow clones)
                - include_history: Whether to include commit history
                
        Returns:
            dict: Processed repository data
        """
        # Validate repository URL
        try:
            parsed = git.Repo.clone_from(repo_url, self._get_temp_dir(), **options)
        except Exception as e:
            raise ProcessingError(f"Failed to clone repository: {e}")
        
        repo_path = Path(self.temp_dir)
        repo = git.Repo(repo_path)
        
        # Checkout specific branch if requested
        if options.get("branch"):
            try:
                repo.git.checkout(options["branch"])
            except Exception as e:
                self.logger.warning(f"Failed to checkout branch {options['branch']}: {e}")
        
        # Extract repository information
        repo_info = self.get_repository_info(repo_url, repo)
        
        # Process code files
        code_files = self.extract_code_files(repo_path, **options.get("file_filters", {}))
        
        # Analyze commits if requested
        commits = []
        if options.get("include_history", True):
            commits = self.analyze_commits(repo_path, **options.get("commit_filters", {}))
        
        # Analyze structure
        structure = self.analyzer.analyze_structure(repo_path)
        metrics = self.analyzer.calculate_metrics(repo_path)
        
        return {
            "repository_info": repo_info,
            "code_files": [f.__dict__ for f in code_files],
            "commits": [c.__dict__ for c in commits],
            "structure": structure,
            "metrics": metrics,
            "temp_path": str(repo_path)
        }
    
    def analyze_commits(self, repo_path: Path, **filters) -> List[CommitInfo]:
        """
        Analyze commit history and patterns.
        
        Args:
            repo_path: Path to repository
            **filters: Filtering criteria:
                - since: Date to start from
                - until: Date to end at
                - author: Filter by author
                - max_commits: Maximum number of commits
                
        Returns:
            list: Commit analysis results
        """
        try:
            repo = git.Repo(repo_path)
        except Exception as e:
            raise ProcessingError(f"Failed to open repository: {e}")
        
        commits = []
        
        # Build git log arguments
        log_args = []
        if filters.get("since"):
            log_args.append(f'--since={filters["since"]}')
        if filters.get("until"):
            log_args.append(f'--until={filters["until"]}')
        if filters.get("author"):
            log_args.append(f'--author={filters["author"]}')
        if filters.get("max_commits"):
            log_args.append(f'-{filters["max_commits"]}')
        
        try:
            for commit in repo.iter_commits(*log_args):
                # Get files changed
                files_changed = [item.a_path for item in commit.stats.files.keys()]
                
                # Calculate stats
                stats = commit.stats.total
                
                commit_info = CommitInfo(
                    hash=commit.hexsha[:7],
                    message=commit.message.strip(),
                    author=str(commit.author),
                    date=datetime.fromtimestamp(commit.committed_date),
                    files_changed=files_changed,
                    additions=stats.get("insertions", 0),
                    deletions=stats.get("deletions", 0)
                )
                commits.append(commit_info)
        except Exception as e:
            self.logger.error(f"Error analyzing commits: {e}")
        
        return commits
    
    def extract_code_files(self, repo_path: Path, **filters) -> List[CodeFile]:
        """
        Extract and process code files from repository.
        
        Args:
            repo_path: Path to repository
            **filters: File filtering criteria:
                - extensions: List of allowed extensions
                - languages: List of allowed languages
                - exclude_patterns: Patterns to exclude
                - max_size: Maximum file size
                
        Returns:
            list: Processed code files
        """
        code_files = []
        
        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Apply filters
            if filters.get("extensions"):
                if file_path.suffix not in filters["extensions"]:
                    continue
            
            if filters.get("exclude_patterns"):
                if any(re.match(pattern, str(file_path)) for pattern in filters["exclude_patterns"]):
                    continue
            
            if filters.get("max_size"):
                if file_path.stat().st_size > filters["max_size"]:
                    continue
            
            # Skip common non-code files
            if file_path.name in ['.gitignore', '.gitkeep', '.gitattributes']:
                continue
            
            if file_path.suffix in ['.pyc', '.pyo', '.pyd', '.so', '.dll']:
                continue
            
            try:
                code_file = self.code_extractor.extract_file_content(file_path)
                
                # Apply language filter
                if filters.get("languages"):
                    if code_file.language not in filters["languages"]:
                        continue
                
                code_files.append(code_file)
            except Exception as e:
                self.logger.debug(f"Failed to extract {file_path}: {e}")
        
        return code_files
    
    def get_repository_info(self, repo_url: str, repo: Optional[git.Repo] = None) -> Dict[str, Any]:
        """
        Get repository metadata and information.
        
        Args:
            repo_url: Repository URL
            repo: Git repository object (optional)
            
        Returns:
            dict: Repository information
        """
        info = {
            "url": repo_url,
            "branches": [],
            "tags": [],
            "remote_info": {}
        }
        
        if repo:
            try:
                info["branches"] = [branch.name for branch in repo.branches]
                info["tags"] = [tag.name for tag in repo.tags]
                
                # Get remote information
                if repo.remotes:
                    remote = repo.remotes.origin
                    info["remote_info"] = {
                        "url": remote.url,
                        "name": remote.name
                    }
                
                # Get latest commit
                if repo.heads:
                    latest_commit = repo.head.commit
                    info["latest_commit"] = {
                        "hash": latest_commit.hexsha[:7],
                        "message": latest_commit.message.strip(),
                        "author": str(latest_commit.author),
                        "date": datetime.fromtimestamp(latest_commit.committed_date).isoformat()
                    }
            except Exception as e:
                self.logger.warning(f"Error getting repository info: {e}")
        
        return info
    
    def _get_temp_dir(self) -> str:
        """Get or create temporary directory for cloning."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="semantica_repo_")
        return self.temp_dir
    
    def cleanup(self):
        """Cleanup temporary repository files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
