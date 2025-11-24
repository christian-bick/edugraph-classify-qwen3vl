def custom_data_collator(batch, processor):
    texts = []
    images = []

    for example in batch:
        # We need to construct the chat from the 'messages' and 'image' columns
        messages = example['messages']
        pil_image = example['image'].convert("RGB")
        
        images.append(pil_image)

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Tokenize the texts and process the images
    batch_data = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    print(f"[Loading Image Data] {batch_data['pixel_values'].shape if 'pixel_values' in batch_data else 'Not Found'}")
    
    # The labels are the input_ids, we need to mask the prompt tokens
    labels = batch_data["input_ids"].clone()
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_data["labels"] = labels
    return batch_data
