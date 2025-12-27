-- ============================================================================
-- NeuronDB pgbench Benchmark: Multimodal Embedding Operations
-- ============================================================================
-- This file benchmarks multimodal embedding operations including:
-- - CLIP embeddings (text and image)
-- - ImageBind embeddings
-- - Cross-modal search
-- ============================================================================

-- Setup: Random values for multimodal operations
\set text_id random(1, 100)
\set modality_id random(0, 1)

-- Benchmark 1: CLIP Text Embedding
SELECT clip_embed(
    CASE :text_id % 5
        WHEN 0 THEN 'A red car driving on a highway'
        WHEN 1 THEN 'A cat sitting on a windowsill'
        WHEN 2 THEN 'A sunset over the ocean'
        WHEN 3 THEN 'A person reading a book in a library'
        ELSE 'A city skyline at night'
    END,
    'text'
) AS clip_text_embedding;

-- Benchmark 2: CLIP Embedding with Different Modality
SELECT clip_embed(
    CASE :text_id % 5
        WHEN 0 THEN 'A red car driving on a highway'
        WHEN 1 THEN 'A cat sitting on a windowsill'
        WHEN 2 THEN 'A sunset over the ocean'
        WHEN 3 THEN 'A person reading a book in a library'
        ELSE 'A city skyline at night'
    END,
    CASE :modality_id WHEN 0 THEN 'text' ELSE 'image' END
) AS clip_multimodal_embedding;

-- Benchmark 3: ImageBind Text Embedding
SELECT imagebind_embed(
    CASE :text_id % 5
        WHEN 0 THEN 'A red car driving on a highway'
        WHEN 1 THEN 'A cat sitting on a windowsill'
        WHEN 2 THEN 'A sunset over the ocean'
        WHEN 3 THEN 'A person reading a book in a library'
        ELSE 'A city skyline at night'
    END,
    'text'
) AS imagebind_text_embedding;

-- Benchmark 4: ImageBind Audio Embedding (text description)
SELECT imagebind_embed(
    CASE :text_id % 5
        WHEN 0 THEN 'A red car driving on a highway'
        WHEN 1 THEN 'A cat sitting on a windowsill'
        WHEN 2 THEN 'A sunset over the ocean'
        WHEN 3 THEN 'A person reading a book in a library'
        ELSE 'A city skyline at night'
    END,
    'audio'
) AS imagebind_audio_embedding;

-- Benchmark 5: ImageBind Video Embedding (text description)
SELECT imagebind_embed(
    CASE :text_id % 5
        WHEN 0 THEN 'A red car driving on a highway'
        WHEN 1 THEN 'A cat sitting on a windowsill'
        WHEN 2 THEN 'A sunset over the ocean'
        WHEN 3 THEN 'A person reading a book in a library'
        ELSE 'A city skyline at night'
    END,
    'video'
) AS imagebind_video_embedding;

-- Benchmark 6: Multimodal Embedding (text + image placeholder)
-- Note: This requires actual image data, so we use a placeholder
SELECT embed_multimodal(
    CASE :text_id % 5
        WHEN 0 THEN 'A red car driving on a highway'
        WHEN 1 THEN 'A cat sitting on a windowsill'
        WHEN 2 THEN 'A sunset over the ocean'
        WHEN 3 THEN 'A person reading a book in a library'
        ELSE 'A city skyline at night'
    END,
    '\x00000000'::bytea,  -- Placeholder image data
    'clip'
) AS multimodal_embedding;

-- Benchmark 7: Cross-Modal Distance (text to text similarity)
SELECT vector_cosine_distance(
    clip_embed(
        CASE :text_id % 5
            WHEN 0 THEN 'A red car driving on a highway'
            WHEN 1 THEN 'A cat sitting on a windowsill'
            WHEN 2 THEN 'A sunset over the ocean'
            WHEN 3 THEN 'A person reading a book in a library'
            ELSE 'A city skyline at night'
        END,
        'text'
    ),
    clip_embed(
        CASE (:text_id + 1) % 5
            WHEN 0 THEN 'A red car driving on a highway'
            WHEN 1 THEN 'A cat sitting on a windowsill'
            WHEN 2 THEN 'A sunset over the ocean'
            WHEN 3 THEN 'A person reading a book in a library'
            ELSE 'A city skyline at night'
        END,
        'text'
    )
) AS cross_modal_distance;

