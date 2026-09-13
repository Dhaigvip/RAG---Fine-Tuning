# Docker Compose Cheatsheet

Docker Compose lets you define multi-container applications in a single YAML file instead of running a long chain of `docker run` commands. The core file is `docker-compose.yml`, which declares services, networks, and volumes.

Common commands worth memorizing: `docker compose up -d` starts everything in detached mode, `docker compose down` stops and removes containers (add `-v` to also drop volumes), `docker compose logs -f <service>` tails logs for one service, and `docker compose exec <service> bash` drops you into a running container's shell.

A gotcha that costs people time: Compose caches the build context by default, so after changing a Dockerfile you often need `docker compose up -d --build` rather than a plain `up`, or the old image just gets reused silently.

Environment variables can come from a `.env` file in the same directory as the compose file, referenced as `${VAR_NAME}` inside `docker-compose.yml` — useful for keeping secrets and per-environment config out of the file itself.
