-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create table for LangChain SupabaseVectorStore (384 dims for all-MiniLM-L6-v2)
create table if not exists public.documents (
    id bigserial primary key,
    content text,
    metadata jsonb,
    embedding vector(384)
);

-- 3. Create similarity search function required by LangChain
create or replace function public.match_documents (
    query_embedding vector(384),
    match_count int default 5,
    filter jsonb default '{}'::jsonb
)
returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from public.documents
    where metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;