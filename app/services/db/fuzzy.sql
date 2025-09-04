-- Trigram-based similarity
create extension if not exists pg_trgm;

-- Optional: normalize accents for better matching
create extension if not exists unaccent;

create index if not exists companies_name_trgm_idx
  on public.companies using gin (company_name gin_trgm_ops);

create index if not exists companies_nse_symbol_trgm_idx
  on public.companies using gin (nse_symbol gin_trgm_ops);

create index if not exists companies_bse_symbol_trgm_idx
  on public.companies using gin (bse_symbol gin_trgm_ops);


create or replace function public.companies_fuzzy_search(q text, limit_count int default 10)
returns setof public.companies
language sql
stable
set search_path = public
as $$
  select c.*
  from public.companies c
  where
    -- Fuzzy match on company_name
    (unaccent(c.company_name) % unaccent(q))
    or (unaccent(c.company_name) ilike '%' || unaccent(q) || '%')

    -- Fuzzy match on NSE/BSE symbols
    or (unaccent(c.nse_symbol) % unaccent(q))
    or (unaccent(c.bse_symbol) % unaccent(q))

    -- Exact match fallback
    or c.bse_code = q
    or c.isin = q

  order by
    greatest(
      similarity(unaccent(c.company_name), unaccent(q)),
      similarity(unaccent(coalesce(c.nse_symbol, '')), unaccent(q)),
      similarity(unaccent(coalesce(c.bse_symbol, '')), unaccent(q))
    ) desc,
    c.company_name asc
  limit limit_count;
$$;
