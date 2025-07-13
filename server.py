import struct
import requests
import torch
import time

from typing import Optional, Tuple
from flask import Flask, request, make_response
from flask_cors import cross_origin

from edge_arch.edge_PartSR import EdgePartSR
edge_arch = EdgePartSR

from config import config, device_name
device = torch.device(device_name)
source_server = config["source_server"]
app = Flask(__name__)

def check_idx(path: str) -> Optional[Tuple[int, str]]:
    split = path.split('/')
    identifier = split[0]
    filename = split[-1]
    if filename.startswith('bbb'):
        parts = filename.split('_')
        idx = parts[-1][:-4]
        identifier = "_".join(parts[:-1])
        return int(idx), identifier
    elif 'stream0' in filename:
        idx = 0 if 'init' in filename else int(filename.split('-')[-1][:-4])
        return idx, identifier
    return None

@app.route('/<path:path>', methods=['GET'])
@cross_origin()
def index(path):
    # sell 的文件名格式：init-stream0.m4s或者chunk-stream0-00001.m4s
    resp = requests.get(f"{source_server}/{path}", params=request.args)
    file = check_idx(path)
    if file:  # 返回处理后的二进制数据
        idx, identifier = file
        if idx == 0:  # 如果是视频头（第一个分片）
            return app.config['edge'].receive(identifier, resp.content, request.args.to_dict(), True)
        else:
            delay_bg = time.perf_counter()
            processed_data = app.config['edge'].receive(identifier, resp.content, request.args.to_dict(), False)
            delay_time = time.perf_counter() - delay_bg
            processed_data = struct.pack(">f", delay_time) + processed_data
            print(f"###### [idx={idx}] delay time = {delay_time:.3f}s, send size = {len(processed_data)}\n ######")
        flask_response = make_response(processed_data)
        flask_response.headers['Content-Type'] = 'application/octet-stream'
        flask_response.headers['Content-Length'] = len(processed_data)
    else:  # 原封不动返回收到的 response
        flask_response = make_response(resp.content)
        flask_response.headers['Content-Type'] = resp.headers.get('Content-Type', 'application/octet-stream')
        flask_response.headers['Content-Length'] = len(resp.content)
    return flask_response

def main():
    edge = edge_arch()
    app.config['edge'] = edge
    app.run(host='0.0.0.0', port=config['edge_port'])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt")
        exit(1)

