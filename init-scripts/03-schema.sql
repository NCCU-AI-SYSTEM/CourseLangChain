ALTER TABLE public.course ADD COLUMN IF NOT EXISTS embedding vector(768);

ALTER TABLE public.course ADD COLUMN IF NOT EXISTS content_segmented TEXT
    GENERATED ALWAYS AS (
        to_tsvector('jiebacfg',
            COALESCE(name, '') || ' ' ||
            COALESCE(nameen, '') || ' ' ||
            COALESCE(objective, '') || ' ' ||
            COALESCE(syllabus, '') || ' ' ||
            COALESCE(schedule, '')
        )::text
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_course_bm25
    ON public.course
    USING bm25(content_segmented)
    WITH (text_config = 'simple');
