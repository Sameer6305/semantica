
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure we test the local code, not the installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from semantica.semantic_extract.ner_extractor import Entity
from semantica.semantic_extract.relation_extractor import Relation
from semantica.semantic_extract.triplet_extractor import Triplet
from semantica.semantic_extract.methods import extract_entities_llm
# We import providers later inside tests to allow patching

class TestSemanticClasses:
    """Test that core semantic classes do not have hardcoded max lengths."""

    def test_entity_no_max_length(self):
        long_text = "a" * 10000
        entity = Entity(text=long_text, label="TEST", start_char=0, end_char=10000)
        assert entity.text == long_text
        assert len(entity.text) == 10000

    def test_relation_no_max_length(self):
        long_text = "a" * 10000
        e1 = Entity(text="s", label="S", start_char=0, end_char=1)
        e2 = Entity(text="o", label="O", start_char=0, end_char=1)
        relation = Relation(subject=e1, predicate=long_text, object=e2)
        assert relation.predicate == long_text

    def test_triplet_no_max_length(self):
        long_text = "a" * 10000
        triplet = Triplet(subject=long_text, predicate="r", object="t")
        assert triplet.subject == long_text


class TestProviderLimits:
    """Test that providers pass through correct length parameters."""

    def test_openai_max_completion_tokens(self):
        from semantica.semantic_extract.providers import OpenAIProvider
        
        # Patch _init_client to avoid real client creation and import issues
        with patch.object(OpenAIProvider, '_init_client', return_value=None):
            provider = OpenAIProvider(api_key="fake")
            
            # Manually mock client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "result"
            mock_client.chat.completions.create.return_value = mock_response
            provider.client = mock_client
            
            provider.generate("prompt", max_completion_tokens=12345, top_p=0.9)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["max_completion_tokens"] == 12345
            assert call_kwargs["top_p"] == 0.9
            assert "max_tokens" not in call_kwargs

    def test_anthropic_max_tokens_defaults(self):
        from semantica.semantic_extract.providers import AnthropicProvider
        
        with patch.object(AnthropicProvider, '_init_client', return_value=None):
            provider = AnthropicProvider(api_key="fake")
            
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="result")]
            mock_client.messages.create.return_value = mock_response
            provider.client = mock_client
            
            provider.generate("prompt")
            
            # Verify default is 8192 (new limit)
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["max_tokens"] == 8192

            # Test override
            provider.generate("prompt", max_tokens=9999)
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["max_tokens"] == 9999

    def test_groq_max_completion_tokens(self):
        from semantica.semantic_extract.providers import GroqProvider
        
        with patch.object(GroqProvider, '_init_client', return_value=None):
            provider = GroqProvider(api_key="fake")
            
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "result"
            mock_client.chat.completions.create.return_value = mock_response
            provider.client = mock_client
            
            provider.generate("prompt", max_completion_tokens=5000)

            # Verify
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["max_completion_tokens"] == 5000

    def test_gemini_params(self):
        from semantica.semantic_extract.providers import GeminiProvider
        
        with patch.object(GeminiProvider, '_init_client', return_value=None):
            provider = GeminiProvider(api_key="fake")
            
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "result"
            mock_model.generate_content.return_value = mock_response
            provider.client = mock_model
            
            provider.generate("prompt", top_k=10, candidate_count=2)

            # Verify
            call_kwargs = mock_model.generate_content.call_args[1]
            gen_config = call_kwargs["generation_config"]
            assert gen_config["top_k"] == 10
            assert gen_config["candidate_count"] == 2

class TestChunkingDefaults:
    """Test that chunking defaults have been increased."""

    @patch("semantica.semantic_extract.methods.create_provider")
    @patch("semantica.semantic_extract.methods._extract_entities_chunked")
    def test_openai_chunking_limit(self, mock_chunked, mock_create_provider):
        # Setup
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_create_provider.return_value = mock_llm
        
        # Text length = 10000 (Greater than old 4000, less than new 64000)
        long_text = "a" * 10000
        
        # Call without explicit max_text_length
        extract_entities_llm(long_text, provider="openai", api_key="fake")
        
        # Should NOT call chunked extraction because default is now 64000
        mock_chunked.assert_not_called()

    @patch("semantica.semantic_extract.methods.create_provider")
    @patch("semantica.semantic_extract.methods._extract_entities_chunked")
    def test_groq_chunking_limit(self, mock_chunked, mock_create_provider):
        # Setup
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_create_provider.return_value = mock_llm
        
        # Text length = 10000 (Greater than old 8000, less than new 64000)
        long_text = "a" * 10000
        
        extract_entities_llm(long_text, provider="groq", api_key="fake")
        
        # Should NOT call chunked extraction because default is now 64000
        mock_chunked.assert_not_called()
