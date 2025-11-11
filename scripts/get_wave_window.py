#     found = None
#     for day_offset in num_offset(max_days):
#         day = center_date + timedelta(days=day_offset)
#         center_date = datetime(2022, 12, 31) if center_date < datetime(2022, 10, 31) else center_date
#         t0 = datetime(day.year, day.month, day.day, 0, 0, 0)
#         t1 = datetime(day.year, day.month, day.day, 23, 59, 59)
#         print(f"Checking {t0:%Y-%m-%d} (offset {day_offset:+d} days) ...")
#         subset_ds = ds.sel(
#             {
#                 "time": slice(t0, t1),
#                 "latitude": slice(lat_min, lat_max),
#                 "longitude": slice(lon_min, lon_max)
#             }
#         )
#         hs = subset_ds["VHM0"]
#         hs_max = hs.max(dim=("latitude", "longitude"))
#         print(hs_max.values)