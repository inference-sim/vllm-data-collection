# Scenario 1

## OpenShift configuration


1. Create new ns
2. `oc adm policy add-scc-to-user anyuid -z default` to allow containers to run as root, required for vLLM images
3. Delete all jobs: `oc delete job --all`