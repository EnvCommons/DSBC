Implementation of [DSBC](https://arxiv.org/pdf/2507.23336).

Changes made compared to original:
- Changed task so LLM-as-a-judge isn't required, answers can be programmatically checked
- Removed the task "What was the net rainfall in March 2023 accord" because it had no solution code


```bash
docker build \
    --platform=linux/amd64 \
    -f environments/environments/dsbc/Dockerfile \
    -t dsbc .
docker tag dsbc us-central1-docker.pkg.dev/indigo-idea-457514-b5/environments/dsbc:8b1218774fca92f3c8c2410b9dbbf4271e55cc84
docker push us-central1-docker.pkg.dev/indigo-idea-457514-b5/environments/dsbc:8b1218774fca92f3c8c2410b9dbbf4271e55cc84

docker build \
    --platform=linux/amd64 \
    -f environments/environments/dsbc/computer.dockerfile \
    -t dsbc-agent .
docker tag dsbc-agent us-central1-docker.pkg.dev/indigo-idea-457514-b5/environments/dsbc-agent:8b1218774fca92f3c8c2410b9dbbf4271e55cc84
docker push us-central1-docker.pkg.dev/indigo-idea-457514-b5/environments/dsbc-agent:8b1218774fca92f3c8c2410b9dbbf4271e55cc84
```