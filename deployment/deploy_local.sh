#!/usr/bin/env bash
# Build the image and deploy to a local Kubernetes cluster
# (Docker Desktop Kubernetes, minikube, or kind).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=heart-disease-api:latest
CTX=$(kubectl config current-context)
echo "Kubernetes context: $CTX"

if [[ "$CTX" == minikube ]]; then
  eval "$(minikube docker-env)"
fi
docker build -t "$IMAGE" .
if [[ "$CTX" == kind-* ]]; then
  kind load docker-image "$IMAGE" --name "${CTX#kind-}"
fi

kubectl apply -f deployment/namespace.yaml
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl rollout status deployment/heart-disease-api -n mlops --timeout=180s
kubectl get pods,svc -n mlops

echo
echo "Port-forwarding to http://localhost:8080 (Ctrl+C to stop)"
kubectl port-forward -n mlops svc/heart-disease-api 8080:80
