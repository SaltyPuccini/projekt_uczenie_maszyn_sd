import psutil


def get_cpu_metrics():
    cpu_times = psutil.cpu_times_percent(interval=None)
    cpu_percent = psutil.cpu_percent(interval=None)
    return {
        'cpu_percent': cpu_percent,
        'cpu_user': cpu_times.user,
        'cpu_system': cpu_times.system,
        'cpu_idle': cpu_times.idle
    }
