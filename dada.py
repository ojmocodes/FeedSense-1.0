from datetime import datetime, timedelta

now = datetime.now()
print(now)
now = str(now).split()[0]
now = f"{now[-2:]}_{now[-5:-3]}_{now[:4]}"

print(now)