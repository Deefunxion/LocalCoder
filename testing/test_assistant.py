# test_assistant.py
# Quick test script for the Academicon Assistant

from main import AcademiconAssistant

# Test queries about Academicon
TEST_QUERIES = [
    "What is the CIP service in Academicon?",
    "How does user authentication work?",
    "What is the structure of the backend API?",
    "Show me the database models for publications",
    "How are citations handled in the system?"
]

def main():
    print("="*60)
    print("Academicon Assistant - Quick Test")
    print("="*60)

    try:
        # Initialize assistant
        assistant = AcademiconAssistant()

        print("\nRunning test queries...\n")

        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(TEST_QUERIES)}")
            print(f"{'='*60}")

            answer = assistant.query(query, verbose=True)

            print(f"\nAnswer Preview (first 300 chars):")
            print(answer[:300] + "..." if len(answer) > 300 else answer)
            print("\n" + "-"*60)

            # Pause between queries
            input("\nPress Enter to continue to next query...")

        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)

    except FileNotFoundError:
        print("\n[ERROR] Vector database not found!")
        print("Please run 'index_academicon.py' first to create the database.")
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
