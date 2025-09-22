import requests
import click
import subprocess
import json
from kubernetes import client, config
from importlib.metadata import version

DOCKERHUB_USER = "oasees"
DOCKERHUB_URL = "https://hub.docker.com/v2"

def check_helm_updates():
    """Check for Helm chart updates by updating repos and comparing versions."""
    try:
        # Update helm repositories
        subprocess.run(["helm", "repo", "update", "oasees-charts"], capture_output=True, check=True)

        # Get installed charts
        result = subprocess.run(
            ["helm", "list", "-o", "json"],
            capture_output=True, text=True, check=True
        )
        installed_charts = json.loads(result.stdout)

        outdated_charts = []

        for chart in installed_charts:
            chart_name,current_version = chart["chart"].rsplit("-",1)  # Remove version suffix


            # Search for latest version
            try:
                search_result = subprocess.run(
                    ["helm", "search", "repo", f"oasees-charts/{chart_name}", "-o", "json"],
                    capture_output=True, text=True, check=True
                )
                search_data = json.loads(search_result.stdout)
                # print(search_data)
                if search_data:
                    latest_version = search_data[0]["version"]

                    if current_version < latest_version:
                        outdated_charts.append({
                            "name": chart["name"],
                            "namespace": chart["namespace"],
                            "chart": chart_name,
                            "current": current_version,
                            "latest": latest_version
                        })
            except subprocess.CalledProcessError:
                continue

        return outdated_charts

    except subprocess.CalledProcessError:
        return []

def get_latest_digests(user):
    """Fetch the latest image digests for all repos under a Docker Hub user/org."""
    repos = []
    url = f"{DOCKERHUB_URL}/repositories/{user}/?page_size=100"
    while url:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        repos.extend([repo["name"] for repo in data["results"]])
        url = data.get("next")

    digests = {}
    for repo in repos:
        url = f"{DOCKERHUB_URL}/repositories/{user}/{repo}/tags/latest"
        r = requests.get(url)
        if r.status_code != 200:
            continue
        data = r.json()
        # Use the tag digest (manifest digest) instead of architecture-specific image digest
        if data.get("digest"):
            digest = data["digest"]
            digests[f"docker.io/{user}/{repo}"] = digest
    return digests


def get_running_digests(namespace="default"):
    """Get running pod image digests from Kubernetes."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod("default")
    running = {}
    for pod in pods.items:
        indexes = []
        for idx,c in enumerate(pod.spec.containers):
            if c.image_pull_policy == 'Always':
                indexes.append(idx)

        for idx in indexes:
            c = pod.status.container_statuses[idx]
            if c.image_id and "@sha256:" in c.image_id:
                repo = c.image.split(":")[0]  # strip tag
                repo_path = repo.split("/")
                if repo_path[0] == "docker.io" and repo_path[1] == "oasees":
                    digest = c.image_id.split("@")[-1]
                    running.setdefault(repo, []).append((pod, c.name, digest))
    return running

# def get_sth():
#     i = 0
#     j = 0
#     config.load_kube_config()
#     v1 = client.CoreV1Api()
#     pods = v1.list_namespaced_pod("default")
#     running = {}
#     for pod in pods.items:
#         indexes = []
#         for idx,c in enumerate(pod.spec.containers):
#             if c.image_pull_policy == 'Always':
#                 indexes.append(idx)

#         for idx in indexes:
#             c = pod.status.container_statuses[idx]
#             if c.image_id and "@sha256:" in c.image_id:
#                 repo = c.image.split(":")[0]  # strip tag
#                 repo_path = repo.split("/")
#                 if repo_path[0] == "docker.io" and repo_path[1] == "oasees":
#                     digest = c.image_id.split("@")[-1]
#                     running.setdefault(repo, []).append(( c.name, digest))

#     print (running)

def delete_pod(pod, namespace="default"):
    """Delete a pod so its controller replaces it with a new one."""
    v1 = client.CoreV1Api()
    v1.delete_namespaced_pod(pod.metadata.name, namespace)
    click.echo(f"Deleted pod {namespace}/{pod.metadata.name} to trigger update")

def update_chart(chart, namespace="default"):
    """Update a Helm chart in the specified namespace."""

    chart_repo_name = chart["chart"]
    chart_local_name = chart["name"]
    try:
        subprocess.run(["helm", "upgrade", chart_local_name, f"oasees-charts/{chart_repo_name}", "--namespace", namespace], capture_output=True, check=True)
        click.secho(f"Updated Helm chart {chart_repo_name} in namespace {namespace}.",fg="green")
    except subprocess.CalledProcessError:
        click.secho(f"Failed to update Helm chart {chart_local_name} in namespace {namespace}.",fg="red")


@click.command()
@click.option("--namespace", default="default", help="Kubernetes namespace to check")
@click.option("--user", default=DOCKERHUB_USER, help="DockerHub username/org")
def update(namespace, user):
    current_version = version("oasees-sdk")

    resp = requests.get("https://pypi.org/pypi/oasees-sdk/json", timeout=3)
    latest_version = resp.json()["info"]["version"]

    if current_version != latest_version:
        click.secho(f"⚠️  A newer version of the oasees-sdk is available ({latest_version}). "
                    f"Please run \"pip install -U oasees-sdk\" to upgrade.", fg="yellow")
        
    else:
        click.secho("The oasees-sdk is up to date.",fg="green")

    click.echo()

    # Check for Helm chart updates
    outdated_helm = check_helm_updates()
    if outdated_helm:
        click.echo("⚠️  Outdated Helm charts detected:\n")
        for chart in outdated_helm:
            click.echo(
                f"- Chart {chart['namespace']}/{chart['name']} "
                f"({chart['chart']})\n"
                f"  Current: {chart['current']}\n"
                f"  Latest : {chart['latest']}\n"
            )
        click.echo()

        if click.confirm("Do you want to update to the latest OASEES Helm charts?"):
            for chart in outdated_helm:
                update_chart(chart, namespace)

    else:
        click.secho("All helm charts are up to date.",fg="green")


    click.echo()

            

    """Check running pods against Docker Hub latest digests and delete outdated ones after confirmation."""
    docker_digests = get_latest_digests(user)
    running_digests = get_running_digests(namespace)

    outdated = []
    for repo, pod_list in running_digests.items():
        latest_digest = docker_digests.get(repo)
        for pod, cname, running_digest in pod_list:
            if latest_digest and running_digest != latest_digest:
                outdated.append((repo, pod, cname, running_digest, latest_digest))

    if not outdated:
        click.secho("All pods are running the latest images.",fg="green")
        return

    click.echo("⚠️  Outdated images detected:\n")
    for repo, pod, cname, running_digest, latest_digest in outdated:
        click.echo(
            f"- Pod {namespace}/{pod.metadata.name} "
            f"(container={cname}, image={repo})\n"
            f"  Running: {running_digest[:12]}\n"
            f"  Latest : {latest_digest[:12]}\n"
        )

    if click.confirm("Do you want to delete these pods so controllers restart them?"):
        for _, pod, _, _, _ in outdated:
            delete_pod(pod, namespace)
    else:
        click.secho("Aborted. No pods deleted.",fg="red")

