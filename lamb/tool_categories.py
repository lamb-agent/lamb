import typing

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

BANKING_CLIENT: set[typing.Callable] = {
    banking_client.get_balance,
    banking_client.get_iban,
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    banking_client.next_id,
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.update_scheduled_transaction,
}

CALENDAR_CLIENT: set[typing.Callable] = {
    calendar_client.add_calendar_event_participants,
    calendar_client.cancel_calendar_event,
    calendar_client.create_calendar_event,
    calendar_client.get_current_day,
    calendar_client.get_day_calendar_events,
    calendar_client.reschedule_calendar_event,
    calendar_client.search_calendar_events,
}

CLOUD_DRIVE_CLIENT: set[typing.Callable] = {
    cloud_drive_client.append_to_file,
    cloud_drive_client.create_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.get_file_by_id,
    cloud_drive_client.list_files,
    cloud_drive_client.search_files,
    cloud_drive_client.search_files_by_filename,
    cloud_drive_client.share_file,
}

EMAIL_CLIENT: set[typing.Callable] = {
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

FILE_READER: set[typing.Callable] = {
    file_reader.read_file,
}

SLACK_CLIENT: set[typing.Callable] = {
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

TRAVEL_BOOKING_CLIENT: set[typing.Callable] = {
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

USER_ACCOUNT: set[typing.Callable] = {
    user_account.get_user_info,
    user_account.update_password,
    user_account.update_user_info,
}

WEB_CLIENT: set[typing.Callable] = {
    web.get_webpage,
    web.download_file,
    web.post_webpage,
    web.standardize_url,
}

ALL: set[typing.Callable] = (
    BANKING_CLIENT
    | CALENDAR_CLIENT
    | CLOUD_DRIVE_CLIENT
    | EMAIL_CLIENT
    | FILE_READER
    | SLACK_CLIENT
    | TRAVEL_BOOKING_CLIENT
    | USER_ACCOUNT
    | WEB_CLIENT
)


READ_ONLY: set[typing.Callable] = {
    banking_client.get_balance,
    banking_client.get_iban,
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    banking_client.next_id,
    calendar_client.get_current_day,
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
"""Read-only tools cannot harm the system
and are thus safe to use in an untrusted (tainted) context.
However, they can still taint the context or leak information.
"""

STATE_CHANGING: set[typing.Callable] = {
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.update_scheduled_transaction,
    calendar_client.add_calendar_event_participants,
    calendar_client.cancel_calendar_event,
    calendar_client.create_calendar_event,
    calendar_client.reschedule_calendar_event,
    cloud_drive_client.append_to_file,
    cloud_drive_client.create_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.share_file,
    email_client.delete_email,
    email_client.send_email,
    slack.add_user_to_channel,
    slack.send_direct_message,
    slack.send_channel_message,
    slack.invite_user_to_slack,
    slack.remove_user_from_slack,
    user_account.update_password,
    travel_booking_client.reserve_car_rental,
    travel_booking_client.reserve_hotel,
    travel_booking_client.reserve_restaurant,
    user_account.update_user_info,
    web.post_webpage,
    web.download_file,
}
"""Tools that perform state changing actions.
They should only be used in a trusted (non-tainted) context."""

TRUSTED_SOURCE: set[typing.Callable] = {
    banking_client.get_balance,
    banking_client.get_iban,  # assuming IBAN type is verified
    banking_client.next_id,
    calendar_client.get_current_day,
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.update_scheduled_transaction,
    calendar_client.cancel_calendar_event,
    email_client.delete_email,
    slack.add_user_to_channel,
    slack.send_direct_message,
    slack.send_channel_message,
    slack.invite_user_to_slack,
    slack.remove_user_from_slack,
    user_account.update_password,
    web.post_webpage,
    web.download_file,
}
"""Tools that return trusted information.
Basically, they don't return arbitrary strings."""

UNTRUSTED_SOURCE: set[typing.Callable] = {
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    calendar_client.add_calendar_event_participants,
    calendar_client.create_calendar_event,
    calendar_client.get_day_calendar_events,
    calendar_client.reschedule_calendar_event,
    calendar_client.search_calendar_events,
    cloud_drive_client.append_to_file,
    cloud_drive_client.create_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.get_file_by_id,
    cloud_drive_client.list_files,
    cloud_drive_client.share_file,
    cloud_drive_client.search_files,
    cloud_drive_client.search_files_by_filename,
    email_client.get_draft_emails,
    email_client.get_received_emails,
    email_client.get_sent_emails,
    email_client.get_unread_emails,
    email_client.search_emails,
    email_client.search_contacts_by_name,
    email_client.search_contacts_by_email,
    email_client.send_email,
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
    travel_booking_client.reserve_car_rental,
    travel_booking_client.reserve_hotel,
    travel_booking_client.reserve_restaurant,
    user_account.get_user_info,
    user_account.update_user_info,
    web.get_webpage,
    web.standardize_url,
}
"""Tools whose results expose the caller to untrusted information,
thus would taint the model, if revealed."""

# TODO: Do we have any trusted sinks?
TRUSTED_SINK: set[typing.Callable] = {
    slack.invite_user_to_slack
}
"""Tools that must only be called with trusted (non-tainted) information."""
UNTRUSTED_SINK: set[typing.Callable] = ALL - TRUSTED_SINK
"""Tools that can be called with untrusted (tainted) information."""

HIGH_CONF_SOURCE: set[typing.Callable] = {
    banking_client.get_balance,
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.update_scheduled_transaction,
    calendar_client.add_calendar_event_participants,
    calendar_client.create_calendar_event,
    calendar_client.get_day_calendar_events,
    calendar_client.reschedule_calendar_event,
    calendar_client.search_calendar_events,
    cloud_drive_client.append_to_file,
    cloud_drive_client.create_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.get_file_by_id,
    cloud_drive_client.list_files,
    cloud_drive_client.search_files,
    cloud_drive_client.search_files_by_filename,
    cloud_drive_client.share_file,
    email_client.get_draft_emails,
    email_client.get_received_emails,
    email_client.get_sent_emails,
    email_client.get_unread_emails,
    email_client.search_emails,
    email_client.search_contacts_by_name,
    email_client.search_contacts_by_email,
    email_client.send_email,
    file_reader.read_file,
    slack.get_channels,
    slack.get_users_in_channel,
    slack.read_channel_messages,
    slack.read_inbox,
    travel_booking_client.reserve_car_rental,
    travel_booking_client.reserve_hotel,
    travel_booking_client.reserve_restaurant,
    user_account.get_user_info,
    user_account.update_user_info,
}
"""Tools that return high confidentiality (secret) information.
Everything that is private user or company internal information is considered secret."""

LOW_CONF_SOURCE: set[typing.Callable] = {
    banking_client.get_iban,
    banking_client.next_id,
    calendar_client.cancel_calendar_event,
    calendar_client.get_current_day,
    email_client.delete_email,
    slack.add_user_to_channel,
    slack.invite_user_to_slack,
    slack.remove_user_from_slack,
    slack.send_direct_message,
    slack.send_channel_message,
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
    user_account.update_password,
    web.get_webpage,
    web.download_file,
    web.post_webpage,
    web.standardize_url,
}
"""Tools that return low confidentiality (public) information.
This is also the case if the tool returns `None` or a static message."""

HIGH_CONF_SINK: set[typing.Callable] = {
    banking_client.get_balance,
    banking_client.get_iban,
    banking_client.get_most_recent_transactions,
    banking_client.get_scheduled_transactions,
    banking_client.next_id,
    calendar_client.get_current_day,
    calendar_client.get_day_calendar_events,
    calendar_client.search_calendar_events,
    cloud_drive_client.create_file,
    cloud_drive_client.get_file_by_id,
    cloud_drive_client.list_files,
    cloud_drive_client.search_files,
    cloud_drive_client.search_files_by_filename,
    email_client.delete_email,
    email_client.get_draft_emails,
    email_client.get_received_emails,
    email_client.get_sent_emails,
    email_client.get_unread_emails,
    email_client.search_emails,
    email_client.search_contacts_by_name,
    email_client.search_contacts_by_email,
    file_reader.read_file,
    # assuming the user is part of the org
    slack.add_user_to_channel,
    slack.get_channels,
    slack.get_users_in_channel,
    slack.read_channel_messages,
    slack.read_inbox,
    slack.remove_user_from_slack,
    # assuming the user is part of the org
    slack.send_direct_message,
    slack.send_channel_message,
    # The travel booking requests would probably leak information in the real world,
    # but we just query our own DB here.
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
    user_account.get_user_info,
    user_account.update_password,
    user_account.update_user_info,
    web.standardize_url,
}
"""Tools whose usage is internal, thus not observable by the untrusted.
This means no confidential information can be leaked."""

LOW_CONF_SINK: set[typing.Callable] = {
    # In a general domain, even GET queries would be considered a low conf sink
    web.post_webpage,
}
"""Tools whose usage is observable by anyone.
Only public information may be sent."""

ARG_CONF_SINK: set[typing.Callable] = {
    banking_client.schedule_transaction,
    banking_client.send_money,
    banking_client.update_scheduled_transaction,
    calendar_client.add_calendar_event_participants,
    calendar_client.cancel_calendar_event,
    calendar_client.create_calendar_event,
    calendar_client.reschedule_calendar_event,
    # cloud drive visibility depends on who the file is shared with
    cloud_drive_client.append_to_file,
    cloud_drive_client.delete_file,
    cloud_drive_client.share_file,
    email_client.send_email,
    slack.invite_user_to_slack,
    web.download_file,
    web.get_webpage,  # can carry arbitrary information in the URL
}
"""Tools where the sink (and its confidentiality level)
is defined by the arguments of the call.
IFC has to be enforced dynamically based on the arguments."""
