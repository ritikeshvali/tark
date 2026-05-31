ml systems explained by someone who came from backend engineering, not ml.

i've spent the last few years building distributed systems on azure, using services like key vault, blob storage, service bus, fabric, data factory, functions. outside of work i've been reading up on and tinkering with things like kafka, s3, distributed kv stores, raft, the usual. i've also been keeping up with llms and ai in general, adding onto my ml and dl knowledge from college. but most of it was gravitating towards prompting, agent workflows, agent orchestration. i wanted to understand the systems underneath: inference, model serving, llm architectures. so i decided to actually build one.

i started building tark, a model inference serving system, and reading blogs, papers, tweets, watching lectures, digging into backend engineering fundamentals. and the deeper i went, the more something kept clicking.

a lot of this is just knowledge transfer from things i already knew: distributed systems, os, networking, dbms. paged attention is os paging, the kv cache is just caching, continuous batching is request scheduling. once you understand the intuition, the connection is so easy to make. that's what i'm trying to capture here.

i'll be writing about a few topics i had to dig deep into: inference systems, attention, transformers, whatever i'm building or reading at the time. partly in technical terms, partly by really dumbing them down, visualizing them, working through them with dummy values, like those examples in school math textbooks that made complicated theorems actually click.

im hoping these posts serve 3 purposes:

1. help someone like me who's trying to actually understand these systems in depth, and is looking for intuition not just definitions
2. force me to understand these concepts with more clarity by explaining them
3. give me a place to refer back to when i forget this intuition, which happens more than i'd like.
