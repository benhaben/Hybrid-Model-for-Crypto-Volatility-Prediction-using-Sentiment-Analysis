import praw

# 替换为你自己的 Reddit 账号和密码
USERNAME = "benhaben1986"
PASSWORD = "qX_cvA3wW6Gm:hK"

reddit = praw.Reddit(
    client_id="1Efd_8bEKo8FPaRiqKgBig",
    client_secret="qKCqiUguDDz1zEFJvl0m03JEOiH5wg",
    user_agent="sentiment-ai/1.0 by benhaben1986",
    username=USERNAME,
    password=PASSWORD,
)

try:
    print("当前登录用户为：", reddit.user.me())
except Exception as e:
    print("登录失败，错误信息如下：")
    print(e)
