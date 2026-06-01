# references

papers and blog posts i read to learn stuff

## orca: a distributed serving system for transformer-based generative models
- url: https://www.usenix.org/conference/osdi22/presentation/yu , https://www.usenix.org/system/files/osdi22-yu.pdf
- what i read: abstract, section 2 (background), section 3 (challenges + solutions)
- key ideas: iteration-level scheduling, selective batching, initiation vs increment phase
- relevant to: continuous batching

## how continuous batching enables 23x throughput in llm inference
- url: https://www.anyscale.com/blog/continuous-batching-llm-inference
- what i read: continuous batching section, skipped pagedattention
- key ideas: static vs continuous batching, variable output lengths kill gpu utilization
- relevant to: continuous batching
