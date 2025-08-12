import datetime

from kygs.utils.typing import TimeUnit


def datetime_floor(dt: datetime.datetime, unit: TimeUnit = "second") -> datetime.datetime:
    if unit == "year":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif unit == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif unit == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif unit == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    elif unit == "minute":
        return dt.replace(second=0, microsecond=0)
    elif unit == "second":
        return dt.replace(microsecond=0)
    else:
        raise ValueError(f"Unknown unit: {unit}")


def datetime_ceil(dt: datetime.datetime, unit: TimeUnit = "second") -> datetime.datetime:
    floored = datetime_floor(dt, unit)
    if floored == dt:
        return dt
    if unit == "year":
        return floored.replace(year=floored.year + 1)
    elif unit == "month":
        if floored.month == 12:
            return floored.replace(year=floored.year + 1, month=1)
        else:
            return floored.replace(month=floored.month + 1)
    elif unit == "day":
        return floored + datetime.timedelta(days=1)
    elif unit == "hour":
        return floored + datetime.timedelta(hours=1)
    elif unit == "minute":
        return floored + datetime.timedelta(minutes=1)
    elif unit == "second":
        return floored + datetime.timedelta(seconds=1)
    else:
        raise ValueError(f"Unknown unit: {unit}")


def increment_datetime(
    dt: datetime,
    unit: TimeUnit = "day",
    amount: int = 1
) -> datetime:
    if unit == "year":
        try:
            # Handle February 29th for non-leap years
            if dt.month == 2 and dt.day == 29 and not (dt.year + amount) % 4 == 0:
                return dt.replace(year=dt.year + amount, month=3, day=1)
            return dt.replace(year=dt.year + amount)
        except ValueError:
            # Handle cases where the day doesn"t exist in the new month (like day 31 in April)
            return (dt.replace(day=1) + datetime.timedelta(days=31)).replace(year=dt.year + amount, day=dt.day)
    
    elif unit == "month":
        new_month = dt.month + amount
        year_offset = (new_month - 1) // 12
        new_month = (new_month - 1) % 12 + 1
        new_year = dt.year + year_offset
        
        try:
            return dt.replace(year=new_year, month=new_month)
        except ValueError:
            # Handle cases where the day doesn"t exist in the new month
            # Move to the first day of next month and subtract one day
            if new_month == 12:
                next_month = 1
                next_year = new_year + 1
            else:
                next_month = new_month + 1
                next_year = new_year
            return datetime(next_year, next_month, 1) - datetime.timedelta(days=1)
    
    elif unit == "day":
        return dt + datetime.timedelta(days=amount)
    elif unit == "hour":
        return dt + datetime.timedelta(hours=amount)
    elif unit == "minute":
        return dt + datetime.timedelta(minutes=amount)
    elif unit == "second":
        return dt + datetime.timedelta(seconds=amount)
    else:
        raise ValueError(f"Unknown unit: {unit}")

