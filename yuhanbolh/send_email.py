import smtplib
from email.mime.text import MIMEText

# 邮件发送文件

# 批量自动化给订阅用户发送邮件，参数：发件人邮箱、发件人邮箱密码、收件人邮箱、邮件内容
def send_email(sender_email, sender_password, receiver_email, error_message):
    message = MIMEText(error_message)
    message["Subject"] = "代码操作通知"  # 邮件主题
    message["From"] = sender_email
    message["To"] = receiver_email

    try:
        # 发送邮件
        with smtplib.SMTP_SSL('smtp.exmail.qq.com', 465) as server:
            server.login(sender_email, sender_password)  # 发件人邮箱用户名和密码
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print("邮件发送失败:", str(e))