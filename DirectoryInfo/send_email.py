import smtplib

# creates SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)

# start TLS for security
s.starttls()

# Authentication
s.login("liila.cheshmi@gmail.com", "1919@music")

# message to be sent
message = "hello leila"

# sending the mail
s.sendmail("liila.cheshmi@gmail.com", "leila.cheshmi@gmail.com", message)

# terminating the session
s.quit()