# # Plotting the volume profile
# plt.figure(figsize=(10,5))
# plt.barh(volume_profile.index, volume_profile.values, height=0.5)
# plt.xlabel("Volume")
# plt.ylabel("Price")
# plt.title("Volume Profile")
# plt.grid(True)
# plt.show()


## narrow range
# Get the opening price for the day
# open_price = yf.download('AMD', start='2023-06-06', end='2023-06-07')
# open_price = open_price.iloc[0, 0]
# print("value: ", open_price)

# # Calculate the 10% range
# lower_bound = open_price * 0.9
# upper_bound = open_price * 1.1

# print("Upper bound: ", upper_bound)
# print("Lower Bound: ", lower_bound)

# ideas:
# show age of point level by opacity