import json

from scripts.data_processing import process_conversation_entry, PROMPT_TEXT, load_and_format_dataset

# Assuming train_dataset.jsonl is in the root of the project
DATASET_PATH = "train_dataset.jsonl"


def test_process_conversation_entry_valid_data():
    """
    Tests process_conversation_entry with valid data from train_dataset.jsonl.
    """
    with open(DATASET_PATH, 'r') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)

            # Process the entry
            processed_messages = process_conversation_entry(entry)

            # Assertions for valid data
            assert isinstance(processed_messages, list)
            assert len(processed_messages) == 3  # system, user, assistant

            # Check system message
            assert processed_messages[0]["role"] == "system"
            assert processed_messages[0]["content"] == PROMPT_TEXT

            # Check user message
            assert processed_messages[1]["role"] == "user"
            assert processed_messages[1]["content"] == [{"type": "image"}]

            # Check assistant message
            assert processed_messages[2]["role"] == "assistant"
            # The content should be a string, which is the original GPT value
            assert isinstance(processed_messages[2]["content"], str)
            assert processed_messages[2]["content"] == entry["conversations"][1]["value"]


def test_process_conversation_entry_invalid_conversations_type():
    """
    Tests process_conversation_entry with an invalid 'conversations' type.
    """
    entry = {"id": "test_id", "image": "test_image.png", "conversations": "not a list"}
    processed_messages = process_conversation_entry(entry)
    assert processed_messages == []


def test_process_conversation_entry_missing_conversations():
    """
    Tests process_conversation_entry with missing 'conversations' key.
    """
    entry = {"id": "test_id", "image": "test_image.png"}
    processed_messages = process_conversation_entry(entry)
    assert processed_messages == []


def test_process_conversation_entry_insufficient_conversations():
    """
    Tests process_conversation_entry with less than 2 conversation turns.
    """
    entry = {"id": "test_id", "image": "test_image.png", "conversations": [{"from": "human", "value": "hi"}]}
    processed_messages = process_conversation_entry(entry)
    assert processed_messages == []


def test_process_conversation_entry_missing_assistant_value():
    """
    Tests process_conversation_entry with missing 'value' in assistant's turn.
    """
    entry = {"id": "test_id", "image": "test_image.png", "conversations": [
        {"from": "human", "value": "human_value"},
        {"from": "gpt", "no_value_key": "some_content"}
    ]}
    processed_messages = process_conversation_entry(entry)
    assert processed_messages == []


def test_process_conversation_entry_invalid_assistant_value_type():
    """
    Tests process_conversation_entry with invalid type for assistant's 'value'.
    """
    entry = {"id": "test_id", "image": "test_image.png", "conversations": [
        {"from": "human", "value": "human_value"},
        {"from": "gpt", "value": ["not a string"]}
    ]}
    processed_messages = process_conversation_entry(entry)
    assert processed_messages == []


def test_load_and_format_dataset_integration():
    """
    Integration test for load_and_format_dataset.
    """
    processed_dataset = load_and_format_dataset(DATASET_PATH, max_samples=10)

    assert len(processed_dataset) == 10

    # Check the structure of the first example
    first_example = processed_dataset[0]
    assert "image" in first_example
    assert "messages" in first_example

    # Check the messages structure
    messages = first_example["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
