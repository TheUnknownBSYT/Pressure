"""Run all experiments in a separate output directory after building the published model."""
from threadpoolctl import threadpool_limits
from . import diagnostics, forecast, opposition, rank, stakes, venue, reliability, report


def main():
    with threadpool_limits(limits=1):
        for stage in [forecast, stakes, rank, opposition, venue, diagnostics, reliability, report]:
            print('Running', stage.__name__, flush=True)
            stage.run()


if __name__ == '__main__':
    main()
