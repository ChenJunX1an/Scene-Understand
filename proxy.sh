function proxy_on() {
export http_proxy=http://10.134.10.78:7890
export https_proxy=http://10.134.10.78:7890
export no_proxy=127.0.0.1,localhost
export HTTP_PROXY=http://10.134.10.78:7890
export HTTPS_PROXY=http://10.134.10.78:7890
export NO_PROXY=127.0.0.1,localhost
}

function proxy_off(){
  unset http_proxy
  unset https_proxy
  unset no_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset NO_PROXY
}