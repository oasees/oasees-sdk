import click
import subprocess
import json
import os
from pathlib import Path
import requests


@click.group(name='telemetry')
def telemetry_commands():
    '''OASEES Telemetry Management Utilities'''
    pass


def get_thanos_endpoint():
    result = subprocess.run(['kubectl', 'get', 'service', 'thanos-query', '-n', 'default', '-o', 'jsonpath={.spec.clusterIP}'], 
                        capture_output=True, text=True)
    cluster_ip = result.stdout.strip()

    url = f"http://{cluster_ip}:9090/api/v1/query"

    return url


@telemetry_commands.command()
def metrics_index():
    '''List All metric names'''

    metrics = []

    url = get_thanos_endpoint()
    params = {'query': '{__name__=~"oasees_.*"}'}
    response = requests.get(url, params=params)
    data = response.json()

    metric_names = set()
    for result in data['data']['result']:
        metric_name = result['metric'].get('metric_index')
        if metric_name:
            metric_names.add(metric_name)

    for name in sorted(metric_names):
        metrics.append(name)

    for m in metrics:
        print(m)

    return metrics

@telemetry_commands.command()
@click.option('--index', '-i', help='Name of the metric to search for')
def metrics_list(index):
    '''Get All metrics by name'''

    url = get_thanos_endpoint()
    params = {'query': f'{{__name__=~"oasees_.*", metric_index="{index}"}}'}
    response = requests.get(url, params=params)
    data = response.json()

    metric_names = set()
    by_source_dict = {}
    for result in data['data']['result']:
        source = result['metric'].get('source')


        oasees_metric = result['metric'].get('__name__').replace("oasees_","")

        if oasees_metric not in by_source_dict:
            by_source_dict[oasees_metric] = set()

        by_source_dict[oasees_metric].add(source)


        if oasees_metric:
            metric_names.add(oasees_metric)

    for m in by_source_dict:
        print(m,"[",*by_source_dict[m],"]")
