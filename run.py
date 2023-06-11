import os
import webbrowser
from http.server import SimpleHTTPRequestHandler, HTTPServer

# 设置要打开的网页文件路径
webpage_path = os.path.abspath("dist/index.html")

# 设置服务器端口号
port = 8000

# 启动 HTTP 服务器
server_address = ("", port)
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

# 打开网页
webbrowser.open(f"http://localhost:{port}")

# 启动服务器并保持运行，直到按下 CTRL-C
print(f"Serving at http://localhost:{port}")
httpd.serve_forever()
