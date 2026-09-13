# System Design Notes

## Load Balancing

### Layer 4 vs Layer 7

Load balancers generally operate at one of two layers of the network stack, and the distinction matters a lot for what kind of routing decisions are possible. A Layer 4 (transport layer) load balancer works purely with IP addresses and TCP/UDP ports — it does not look inside the packet payload at all. This makes it extremely fast, since there's no need to parse or understand the application protocol, but it also means routing decisions can only be based on things like source IP, destination IP, and port number. A Layer 4 balancer typically uses a hashing scheme (source IP hash, or a simple round robin) to pick a backend, and it maintains that mapping for the life of the connection so that all packets in a TCP session go to the same backend.

A Layer 7 (application layer) load balancer, by contrast, terminates the connection and actually reads the HTTP request — the path, the headers, cookies, even the body in some cases. This lets it make much smarter routing decisions: it can route `/api/v2/*` to one backend fleet and `/static/*` to another, it can do session affinity based on a cookie rather than just source IP, and it can inspect the `Host` header to support many different domains behind one balancer (this is how most CDNs and API gateways work). The cost is that Layer 7 balancing requires terminating and re-establishing TCP connections (or at least fully parsing the HTTP stream), which adds CPU overhead and latency compared to Layer 4.

In practice, most cloud load balancer products (AWS ALB, GCP's HTTP(S) load balancer) are Layer 7 by default because the flexibility is worth the overhead for typical web traffic, while Layer 4 products (AWS NLB) get reserved for cases needing extreme throughput, non-HTTP protocols, or where you need to preserve the client's source IP all the way to the backend without proxy rewriting. A useful mental shortcut: if the routing decision needs to know anything about the *content* of the request, you need Layer 7; if it only needs to know *where the packet is going*, Layer 4 is faster and simpler. Understanding which layer a given load balancer operates at also explains a lot of confusing production behavior — for example, why a Layer 4 balancer can't do path-based routing at all, or why TLS termination has to happen somewhere in the Layer 7 case (either at the balancer or passed through to the backend as an encrypted stream, called TLS passthrough, which keeps it functioning more like a Layer 4 device for that specific traffic).

One more wrinkle worth remembering: health checks differ between the two as well. A Layer 4 balancer can only really check "is this port accepting TCP connections," while a Layer 7 balancer can send an actual HTTP request to a health check path and inspect the status code and even response body, which catches a much wider class of failures (an app that accepts connections but returns 500s on every request, for instance, looks perfectly healthy to a Layer 4 check but obviously unhealthy to a Layer 7 one).

## Caching Strategies

### Cache-Aside

The application checks the cache first; on a miss, it reads from the database and writes the result into the cache before returning it. Simple and widely used, but every cache miss pays the full database latency, and there's a window where the cache can be stale relative to the database.

### Write-Through

Writes go to the cache and the database at the same time (or the cache writes through to the database on every write). Keeps the cache always consistent with the database, at the cost of extra write latency on every write, even for data that might never be read again.
