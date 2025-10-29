import re
from kubernetes import client, config
from kubernetes.client import Configuration
from kubernetes.stream import stream
import time

import urllib3
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)

NAMESPACE = "blis"

config.load_kube_config()

c = Configuration().get_default_copy()
c.verify_ssl = False

v1_batch = client.BatchV1Api()
v1_core = client.CoreV1Api()    


try:
    jobs = v1_batch.list_namespaced_job(namespace=NAMESPACE)
    print(f"Active Jobs in namespace '{NAMESPACE}':")
    if not jobs.items:
        print("No jobs found in this namespace.")
    else:
        for job in jobs.items:
            print(f"- {job.metadata.name}")
            if job.status.active:
                selector_labels = job.spec.selector.match_labels
                    
                label_selector_str = ",".join([f"{k}={v}" for k, v in selector_labels.items()])

                # List pods in the namespace that match the job's label selector.
                pods = v1_core.list_namespaced_pod(
                    namespace=NAMESPACE,
                    label_selector=label_selector_str
                )
                
                if pods.items:
                    print("  Associated Pods:")
                    for pod in pods.items:
                        pod_name = pod.metadata.name
                        print (f"        - {pod_name}")
                        pod_status = pod.status.phase
                        if pod_status == "Running":
                            try:
                                # Fetch logs, optionally specifying a container for multi-container pods
                                logs = v1_core.read_namespaced_pod_log(
                                    name=pod_name,
                                    namespace=NAMESPACE,
                                    container="vllm-client",
                                    _preload_content=False
                                ).data.decode('utf-8')
                                
                                # 4. Search logs for the pattern
                                log_pattern = r"sleep 30000000"
                                if re.search(log_pattern, logs):
                                    print(f"\n--- Match found in logs for Job: {job.metadata.name}, Pod: {pod_name} ---")
                                    delete_options = client.V1DeleteOptions(propagation_policy='Foreground')
                                    try:
                                        time.sleep(30)
                                        v1_batch.delete_namespaced_job(name=job.metadata.name, namespace=NAMESPACE, body=delete_options)
                                        print(f"Job '{job.metadata.name}' in namespace '{NAMESPACE}' deleted.")
                                    except client.ApiException as e:
                                        print(f"Error deleting job: {e}")
                                
                            except client.ApiException as e:
                                print(f"Error retrieving logs for pod {pod_name}: {e}")
                else:
                    print(" No pods found for this job.")
except client.ApiException as e:
    print(f"Error listing jobs: {e}")
