from datetime import datetime
from dateutil.relativedelta import relativedelta

# Function to calculate months since the last donation and check the gap
def check_donation_gap(last_donation_date, number_of_donations):
    current_date = datetime.now()
    
    # Parse the last donation date (assuming it is in the format YYYY-MM-DD)
    last_donation = datetime.strptime(last_donation_date, "%Y-%m-%d")
    
    # Calculate the difference between the current date and the last donation date
    difference = relativedelta(current_date, last_donation)
    
    # Get the number of months since last donation
    months_since_last_donation = difference.years * 12 + difference.months
    
    # Check if the person has donated more than 6 times in the year
    if number_of_donations >= 6:
        return f"You have exceeded the maximum number of donations (6 per year). You cannot donate blood."
    
    # Check if the person needs to wait 2 months or more for the next donation
    if months_since_last_donation >= 2:
        return f"You can donate now. Months since last donation: {months_since_last_donation} months."
    else:
        return f"You need to wait. Months since last donation: {months_since_last_donation} months. You need at least 2 months gap."

# Example usage
last_donation_date = "2024-10-01"  # Replace with the actual date of last donation
number_of_donations = 5  # Replace with the actual number of donations in the current year
result = check_donation_gap(last_donation_date, number_of_donations)
print(result)
