from primus.agent import start, stop
from primus.ui import C, CSS, build

if __name__ == '__main__':
    try:
        start()
        build().launch(server_name='127.0.0.1', server_port=7860, css=CSS)
    finally:
        stop()
        C.close()