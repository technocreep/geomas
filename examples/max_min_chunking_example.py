"""
Test script for Max-Min semantic chunking algorithm.

This script demonstrates how to use the Max-Min chunking algorithm
instead of the standard chunking approach.
"""

from geomas.core.rag_modules.parser.rag_parser import DocumentParser
from geomas.core.rag_modules.steps.max_min_chunking import MaxMinTextChunker, MaxMinChunkingParams


def test_max_min_chunking_simple():
    """Test Max-Min chunking with a simple text"""
    
    print("=" * 80)
    print("Test 1: Simple text chunking with Max-Min algorithm")
    print("=" * 80)
    
    # Sample text
    sample_text = """
    Artificial intelligence is transforming the world. Machine learning is a subset of AI.
    Deep learning uses neural networks. Neural networks mimic the human brain.
    Computer vision enables machines to see. Natural language processing helps machines understand text.
    Robotics combines AI with mechanical engineering. Autonomous vehicles use AI for navigation.
    The future of AI is promising. Many industries are adopting AI solutions.
    """
    
    # Initialize Max-Min chunker with custom parameters
    params = {
        'c': 0.8,
        'hard_thr': 0.5,
        'init_const': 1.2,
        'max_chunk_sentences': 5,
    }
    
    chunker = MaxMinTextChunker(chunking_params=params)
    
    try:
        # Apply chunking
        documents = chunker.apply_chunking(
            raw_text=sample_text,
            document_name="test_document",
            document_type="html"
        )
        
        # Print results
        print(f"\n[SUCCESS] Created {len(documents)} chunks\n")
        
        for i, doc in enumerate(documents):
            print(f"Chunk {i + 1}:")
            print(f"  Sentences: {doc.metadata['sentence_count']}")
            print(f"  Content: {doc.page_content[:100]}...")
            print()
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        print("\n[WARNING] Make sure the embedding service is running!")
        print("   Check EMBEDDING_HOST and EMBEDDING_PORT in your environment.")


def test_document_parser_with_max_min():
    """Test DocumentParser with Max-Min algorithm enabled"""
    
    print("\n" + "=" * 80)
    print("Test 2: DocumentParser with Max-Min chunking")
    print("=" * 80)
    
    # Initialize parser with Max-Min algorithm
    parser = DocumentParser(use_max_min=True)
    
    print(f"\n[SUCCESS] DocumentParser initialized with Max-Min algorithm")
    print(f"   Chunking agent type: {type(parser.chunking_agent).__name__}")


def test_standard_vs_max_min():
    """Compare standard chunking with Max-Min chunking"""
    
    print("\n" + "=" * 80)
    print("Test 3: Comparison of chunking algorithms")
    print("=" * 80)
    
    sample_text = """
    Geology is the study of Earth. Rocks tell stories of ancient times.
    Minerals form crystals underground. Volcanoes shape the landscape.
    Earthquakes move tectonic plates. Mountains rise over millions of years.
    Fossils preserve ancient life. Sedimentary layers record history.
    """
    
    # Standard chunking
    print("\n[STANDARD] Standard chunking:")
    parser_standard = DocumentParser(use_max_min=False)
    print(f"   Using: {type(parser_standard.chunking_agent).__name__}")
    
    # Max-Min chunking
    print("\n[MAX-MIN] Max-Min semantic chunking:")
    parser_maxmin = DocumentParser(use_max_min=True)
    print(f"   Using: {type(parser_maxmin.chunking_agent).__name__}")
    
    print("\n[INFO] To see actual chunking differences, apply both to a real document.")


def main():
    """Run all tests"""
    
    print("\n" + "=" * 80)
    print("Max-Min Semantic Chunking Algorithm Test Suite")
    print("=" * 80 + "\n")
    
    # Test 1: Simple chunking
    test_max_min_chunking_simple()
    
    # Test 2: Parser initialization
    test_document_parser_with_max_min()
    
    # Test 3: Comparison
    test_standard_vs_max_min()
    
    print("\n" + "=" * 80)
    print("[DONE] All tests completed!")
    print("=" * 80)
    
    print("\n[USAGE] Usage examples:")
    print("   1. Standard chunking:  parser = DocumentParser()")
    print("   2. Max-Min chunking:   parser = DocumentParser(use_max_min=True)")
    print("   3. Custom parameters:  chunker = MaxMinTextChunker(chunking_params={...})")
    print("\n")


if __name__ == "__main__":
    main()

