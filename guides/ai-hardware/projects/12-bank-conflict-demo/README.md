# Bank Conflict Demo

## Key Insight

Avoiding [bank conflicts](/shared/glossary/#bank-conflict) is essential for maximizing the speed of [shared memory](/shared/glossary/#shared-memory) accesses on a [GPU](/shared/glossary/#gpu). Because shared memory is divided into 32 distinct memory banks that handle requests in parallel, having multiple threads in a [warp](/shared/glossary/#warp) request different addresses in the same bank forces the hardware to serialize those accesses. Designing data access patterns that distribute thread requests evenly across banks avoids serialization stalls, ensuring near-instantaneous memory retrieval.
