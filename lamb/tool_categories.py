from agentdojo.default_suites.v1.tools import (
    banking_client,
    calendar_client,
    cloud_drive_client,
    email_client,
    file_reader,
    slack,
    travel_booking_client,
    user_account,
    web,
)

BANKING_CLIENT = {
    banking_client.get_balance,
    banking_client.get_iban,
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    banking_client.next_id,
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.set_balance,
    banking_client.set_iban,
    banking_client.update_scheduled_transaction,
}

CALENDAR_CLIENT = {
    calendar_client.add_calendar_event_participants,
    calendar_client.cancel_calendar_event,
    calendar_client.create_calendar_event,
    calendar_client.get_current_day,
    calendar_client.get_day_calendar_events,
    calendar_client.reschedule_calendar_event,
    calendar_client.search_calendar_events,
}

CLOUD_DRIVE_CLIENT = {
    cloud_drive_client.append_to_file,
    cloud_drive_client.create_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.get_file_by_id,
    cloud_drive_client.list_files,
    cloud_drive_client.search_files,
    cloud_drive_client.search_files_by_filename,
    cloud_drive_client.share_file,
}

EMAIL_CLIENT = {
    email_client.delete_email,
    email_client.get_draft_emails,
    email_client.get_received_emails,
    email_client.get_sent_emails,
    email_client.get_unread_emails,
    email_client.search_emails,
    email_client.search_contacts_by_name,
    email_client.search_contacts_by_email,
    email_client.send_email,
}

FILE_READER = {
    file_reader.read_file,
}

SLACK_CLIENT = {
    slack.add_user_to_channel,
    slack.get_channels,
    slack.get_users_in_channel,
    slack.invite_user_to_slack,
    slack.read_channel_messages,
    slack.read_inbox,
    slack.remove_user_from_slack,
    slack.send_direct_message,
    slack.send_channel_message,
}

TRAVEL_BOOKING_CLIENT = {
    travel_booking_client.check_restaurant_opening_hours,
    travel_booking_client.get_all_car_rental_companies_in_city,
    travel_booking_client.get_all_hotels_in_city,
    travel_booking_client.get_all_restaurants_in_city,
    travel_booking_client.get_car_rental_address,
    travel_booking_client.get_car_fuel_options,
    travel_booking_client.get_car_price_per_day,
    travel_booking_client.get_car_types_available,
    travel_booking_client.get_cuisine_type_for_restaurants,
    travel_booking_client.get_contact_information_for_restaurants,
    travel_booking_client.get_dietary_restrictions_for_all_restaurants,
    travel_booking_client.get_flight_information,
    travel_booking_client.get_hotels_prices,
    travel_booking_client.get_hotels_address,
    travel_booking_client.get_price_for_restaurants,
    travel_booking_client.get_rating_reviews_for_car_rental,
    travel_booking_client.get_rating_reviews_for_hotels,
    travel_booking_client.get_rating_reviews_for_restaurants,
    travel_booking_client.get_restaurants_address,
    travel_booking_client.get_user_information,
    travel_booking_client.reserve_car_rental,
    travel_booking_client.reserve_hotel,
    travel_booking_client.reserve_restaurant,
}

USER_ACCOUNT = {
    user_account.get_user_info,
    user_account.update_password,
    user_account.update_user_info,
}

WEB_CLIENT = {
    web.get_webpage,
    web.download_file,
    web.post_webpage,
    web.standardize_url,
}

READ_ONLY_TRUSTED = {
    banking_client.get_balance,
    banking_client.get_iban,  # assuming IBAN type is verified
    banking_client.next_id,
    calendar_client.get_current_day,
}

READ_ONLY_UNTRUSTED = {
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    calendar_client.get_day_calendar_events,
    calendar_client.search_calendar_events,
    cloud_drive_client.get_file_by_id,
    cloud_drive_client.list_files,
    cloud_drive_client.search_files_by_filename,
    cloud_drive_client.search_files,
    email_client.get_draft_emails,
    email_client.get_received_emails,
    email_client.get_sent_emails,
    email_client.get_unread_emails,
    email_client.search_emails,
    email_client.search_contacts_by_name,
    email_client.search_contacts_by_email,
    file_reader.read_file,
    slack.get_channels,
    slack.get_users_in_channel,
    slack.read_channel_messages,
    slack.read_inbox,
    travel_booking_client.check_restaurant_opening_hours,
    travel_booking_client.get_all_car_rental_companies_in_city,
    travel_booking_client.get_all_hotels_in_city,
    travel_booking_client.get_all_restaurants_in_city,
    travel_booking_client.get_car_rental_address,
    travel_booking_client.get_car_price_per_day,
    travel_booking_client.get_car_types_available,
    travel_booking_client.get_cuisine_type_for_restaurants,
    travel_booking_client.get_contact_information_for_restaurants,
    travel_booking_client.get_dietary_restrictions_for_all_restaurants,
    travel_booking_client.get_flight_information,
    travel_booking_client.get_hotels_prices,
    travel_booking_client.get_hotels_address,
    travel_booking_client.get_price_for_restaurants,
    travel_booking_client.get_rating_reviews_for_car_rental,
    travel_booking_client.get_rating_reviews_for_hotels,
    travel_booking_client.get_rating_reviews_for_restaurants,
    travel_booking_client.get_restaurants_address,
    travel_booking_client.get_user_information,
    user_account.get_user_info,
    web.get_webpage,
    web.standardize_url,
}

READ_ONLY = READ_ONLY_TRUSTED | READ_ONLY_UNTRUSTED

STATE_CHANGING_TRUSTED = {
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.set_balance,
    banking_client.set_iban,
    banking_client.update_scheduled_transaction,
    calendar_client.cancel_calendar_event,
    email_client.delete_email,
    email_client.send_email,
    slack.add_user_to_channel,
    slack.send_direct_message,
    slack.send_channel_message,
    slack.invite_user_to_slack,
    slack.remove_user_from_slack,
    user_account.update_password,
    web.post_webpage,
    web.download_file,
}

STATE_CHANGING_UNTRUSTED = {
    calendar_client.add_calendar_event_participants,
    calendar_client.create_calendar_event,
    calendar_client.reschedule_calendar_event,
    cloud_drive_client.append_to_file,
    cloud_drive_client.create_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.share_file,
    travel_booking_client.reserve_car_rental,
    travel_booking_client.reserve_hotel,
    travel_booking_client.reserve_restaurant,
    user_account.update_user_info,
}

STATE_CHANGING = STATE_CHANGING_TRUSTED | STATE_CHANGING_UNTRUSTED

TRUSTED = STATE_CHANGING_TRUSTED | READ_ONLY_TRUSTED
