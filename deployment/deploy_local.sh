#!/usr/bin/env bash
# Build the image and deploy to a local Kubernetes cluster
# (Docker Desktop Kubernetes, minikube, or kind), then verify the endpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=heart-disease-api:1.0.0
CTX=$(kubectl config current-context)
echo "Kubernetes context: $CTX"

if [[ "$CTX" == minikube ]]; then
  eval "$(minikube docker-env)"
fi
docker build -t "$IMAGE" -t heart-disease-api:latest .
if [[ "$CTX" == kind-* ]]; then
  kind load docker-image "$IMAGE" --name "${CTX#kind-}"
fi

kubectl apply -f deployment/namespace.yaml
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
# Ingress is optional: only applied when an ingress controller is installed.
if kubectl get ingressclass nginx >/dev/null 2>&1; then
  kubectl apply -f deployment/ingress.yaml
fi
kubectl rollout restart deployment/heart-disease-api -n mlops
kubectl rollout status deployment/heart-disease-api -n mlops --timeout=240s
kubectl get all -n mlops

URL=${URL:-http://localhost:8080}
echo "Waiting for the LoadBalancer at $URL ..."
for _ in $(seq 1 30); do
  curl -sf "$URL/health" >/dev/null && break
  sleep 2
done
if ! curl -sf "$URL/health" >/dev/null; then
  echo "LoadBalancer not reachable on localhost - falling back to kubectl port-forward"
  kubectl port-forward -n mlops svc/heart-disease-api 8080:8080 >/tmp/hd-port-forward.log 2>&1 &
  sleep 5
fi
echo "--- GET /health";  curl -s "$URL/health"; echo
echo "--- POST /predict"
curl -s -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
echo
echo "Deployed. API: $URL  (docs: $URL/docs, metrics: $URL/metrics)"
