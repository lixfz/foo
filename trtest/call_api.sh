#!/bin/bash
function get_content(){
    while read -r line; do
        # 使用 grep 和 sed 提取 "content" 的值
        content=$(echo "$line" | grep -o '"content":"[^"]*"' | sed 's/"content":"\(.*\)"/\1/')

        # 如果提取到内容，则显示
        if [ -n "$content" ]; then
            echo -n "$content"
        fi
    done
}
        
#pod_ip=$(kubectl -n nccl-tests get po -o wide|grep vds-01-0 |gawk '{print $6}')
pod_addr="127.0.0.1:30000"
echo pod_addr: $pod_addr

time curl -s --location http://$pod_addr/v1/chat/completions \
        --header 'Authorization: Bearer sk_foo' \
        --header 'Content-Type: application/json' \
        --data '{
            "stream": true,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "你是一个特级中学语文老师，特别擅长唐诗的研究，会高度模仿李白、杜甫的作诗风格，还特别能接受别人的意见。帮做仿照李白的风格写一首迎春的7言律诗。"
                }
            ],
            "model": "deepseek-r1"
        }' | get_content

echo ""
