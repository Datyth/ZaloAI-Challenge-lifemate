"""Tasks: hotel_booking, clinic. Time anchor Thursday 2026-11-12 unless stated."""
from _common import ASK, C, CONTACTS, REMINDERS, SMS, CALENDAR, T

NOW = "2026-11-12T09:00:00+07:00"
HOTEL = {"hotels": [
    {"id": "h_01", "name": "Khách sạn Đồi Thông", "city": "Đà Lạt", "address": "12 Trần Hưng Đạo, Phường 3, Đà Lạt", "price_per_night": 850000, "rating": 4.3,
     "rooms": {"standard": 3, "deluxe": 1, "family": 0}},
    {"id": "h_02", "name": "Hoa Hồng Boutique", "city": "Đà Lạt", "address": "5 Hoàng Văn Thụ, Phường 4, Đà Lạt", "price_per_night": 1450000, "rating": 4.7,
     "rooms": {"standard": 2, "deluxe": 2, "family": 1}},
    {"id": "h_03", "name": "Sea Breeze Hotel", "city": "Nha Trang", "address": "88 Trần Phú, Lộc Thọ, Nha Trang", "price_per_night": 1200000, "rating": 4.5,
     "rooms": {"standard": 4, "deluxe": 2, "family": 2}},
    {"id": "h_04", "name": "Khách sạn Phố Cổ", "city": "Hội An", "address": "21 Bạch Đằng, Minh An, Hội An", "price_per_night": 690000, "rating": 4.1,
     "rooms": {"standard": 1, "deluxe": 0, "family": 1}},
], "bookings": [
    {"id": "bk_01", "hotel_id": "h_03", "room_type": "deluxe", "check_in": "2026-11-20", "check_out": "2026-11-22", "guests": 2, "status": "confirmed"},
]}
H = {"hotel_booking": HOTEL}
HS = {"hotel_booking": HOTEL, "contacts": CONTACTS, "sms": SMS}

CLINIC = {"doctors": [
    {"id": "dr_01", "name": "BS. Nguyễn Thu Trang", "specialty": "pediatrics", "clinic": "Phòng khám Nhi An Khang",
     "available_slots": ["2026-11-13T09:00", "2026-11-13T10:30", "2026-11-14T08:30"]},
    {"id": "dr_02", "name": "BS. Lê Minh Quân", "specialty": "dentistry", "clinic": "Nha khoa Sài Gòn Smile",
     "available_slots": ["2026-11-12T15:00", "2026-11-13T17:30", "2026-11-16T18:00"]},
    {"id": "dr_03", "name": "BS. Phạm Hoài Nam", "specialty": "ent", "clinic": "Phòng khám Tai Mũi Họng Hoàn Mỹ",
     "available_slots": ["2026-11-12T16:00", "2026-11-14T09:00"]},
    {"id": "dr_04", "name": "BS. Trần Kim Oanh", "specialty": "dermatology", "clinic": "Da liễu Thiên Phúc",
     "available_slots": ["2026-11-17T14:00", "2026-11-18T10:00"]},
], "appointments": [
    {"id": "apt_01", "doctor_id": "dr_02", "time": "2026-11-12T10:00:00+07:00", "patient": "self", "reason": "lấy cao răng", "status": "booked"},
]}
CL = {"clinic": CLINIC}
CLR = {"clinic": CLINIC, "reminders": REMINDERS}

TASKS = [
    # hotel_booking
    T("hotel_search_book", NOW, "Đặt giúp anh một phòng thường ở Khách sạn Đồi Thông Đà Lạt, nhận phòng thứ bảy này, trả chủ nhật, hai người nhé.",
      [C("hotel_booking.search_hotels", city="Đà Lạt", query="Đồi Thông"),
       C("hotel_booking.book", hotel_id="h_01", room_type="standard", check_in="2026-11-14", check_out="2026-11-15", guests=2)], H),
    T("hotel_cheapest", NOW, "Tìm khách sạn ở Hội An dưới bảy trăm nghìn một đêm rồi đặt phòng gia đình từ hai mươi tới hai hai tháng này cho bốn người.",
      [C("hotel_booking.search_hotels", city="Hội An", max_price=700000),
       C("hotel_booking.book", hotel_id="h_04", room_type="family", check_in="2026-11-20", check_out="2026-11-22", guests=4)], H),
    T("hotel_deluxe_note", NOW, "Book phòng deluxe ở Hoa Hồng Boutique đêm mai cho vợ chồng em, ghi chú là muốn phòng tầng cao nha.",
      [C("hotel_booking.search_hotels", query="Hoa Hồng Boutique"),
       C("hotel_booking.book", hotel_id="h_02", room_type="deluxe", check_in="2026-11-13", check_out="2026-11-14", guests=2, note="muốn phòng tầng cao")], H),
    T("hotel_missing_dates", NOW, "Đặt cho chị một phòng thường ở Sea Breeze Nha Trang nhé.",
      [C("hotel_booking.search_hotels", query="Sea Breeze"), ASK("hotel_booking.book", "check_in", "check_out", "guests")], H),
    T("hotel_missing_guests", NOW, "Đặt phòng family ở Sea Breeze từ hai mươi bốn đến hai mươi sáu tháng mười một.",
      [C("hotel_booking.search_hotels", query="Sea Breeze"), ASK("hotel_booking.book", "guests")], H),
    T("hotel_cancel", NOW, "Huỷ cái đặt phòng Nha Trang tuần sau của anh đi.",
      [C("hotel_booking.list_bookings"), C("hotel_booking.cancel_booking", booking_id="bk_01")], H),
    T("hotel_change", NOW, "Chuyến Nha Trang dời lại một ngày nha, nhận phòng hai mốt trả hai ba.",
      [C("hotel_booking.list_bookings"), C("hotel_booking.change_dates", booking_id="bk_01", check_in="2026-11-21", check_out="2026-11-23")], H),
    T("hotel_book_and_sms", NOW, "Đặt phòng thường Khách sạn Phố Cổ Hội An đêm thứ sáu tuần sau một người, xong nhắn chị Hà là em đặt rồi nhé.",
      [C("hotel_booking.search_hotels", query="Phố Cổ"),
       C("hotel_booking.book", hotel_id="h_04", room_type="standard", check_in="2026-11-20", check_out="2026-11-21", guests=1),
       C("contacts.search", query="Chị Hà"), C("sms.send", contact_id="c_01", content="Em đặt phòng khách sạn Phố Cổ Hội An rồi nhé")], HS),
    T("hotel_conditional", NOW, "Xem Hoa Hồng Boutique còn phòng gia đình không, còn thì đặt cuối tuần này thứ bảy tới chủ nhật cho năm người, hết thì thôi.",
      [C("hotel_booking.search_hotels", query="Hoa Hồng Boutique"),
       C("hotel_booking.book", hotel_id="h_02", room_type="family", check_in="2026-11-14", check_out="2026-11-15", guests=5)], H),
    T("hotel_conditional_none", NOW, "Đồi Thông còn phòng gia đình không em, nếu còn thì đặt tối nay cho bốn người.",
      [C("hotel_booking.search_hotels", query="Đồi Thông")], H),
    T("hotel_list", NOW, "Anh đang có đặt phòng nào sắp tới không?", [C("hotel_booking.list_bookings")], H),
    T("hotel_out_of_scope", NOW, "Đặt giúp tôi vé máy bay đi Đà Lạt sáng thứ bảy.", [], {}, uninstall=["hotel_booking", "flights"]),
    # clinic
    T("clinic_book_kid", NOW, "Đặt lịch khám nhi cho bé Na chín giờ sáng mai nha, bé bị ho.",
      [C("clinic.search_doctors", specialty="pediatrics"), C("clinic.book_appointment", doctor_id="dr_01", time="2026-11-13T09:00:00+07:00", patient="bé Na", reason="bị ho")], CL),
    T("clinic_dentist_today", NOW, "Chiều nay ba giờ còn lịch nha sĩ không, đặt cho tôi luôn.",
      [C("clinic.search_doctors", specialty="dentistry"), C("clinic.book_appointment", doctor_id="dr_02", time="2026-11-12T15:00:00+07:00")], CL),
    T("clinic_ent_named", NOW, "Đặt bác sĩ Phạm Hoài Nam tai mũi họng sáng thứ bảy lúc chín giờ giúp em.",
      [C("clinic.search_doctors", query="Phạm Hoài Nam"), C("clinic.book_appointment", doctor_id="dr_03", time="2026-11-14T09:00:00+07:00")], CL),
    T("clinic_missing_time", NOW, "Đặt lịch khám da liễu cho chị với bác sĩ Kim Oanh.",
      [C("clinic.search_doctors", query="Kim Oanh"), ASK("clinic.book_appointment", "time")], CL),
    T("clinic_reschedule", NOW, "Lịch lấy cao răng sáng nay dời sang năm rưỡi chiều mai được không em.",
      [C("clinic.list_appointments"), C("clinic.reschedule", appointment_id="apt_01", time="2026-11-13T17:30:00+07:00")], CL),
    T("clinic_cancel", NOW, "Hủy lịch nha khoa sáng nay giùm anh.", [C("clinic.list_appointments"), C("clinic.cancel_appointment", appointment_id="apt_01")], CL),
    T("clinic_book_remind", NOW, "Đặt bác sĩ da liễu hai giờ chiều thứ ba tuần sau, rồi nhắc chị trước đó một tiếng nhé.",
      [C("clinic.search_doctors", specialty="dermatology"), C("clinic.book_appointment", doctor_id="dr_04", time="2026-11-17T14:00:00+07:00"),
       C("reminders.create", content="đi khám da liễu", time="2026-11-17T13:00:00+07:00")], CLR),
    T("clinic_which_doctor", NOW, "Đặt lịch khám cho mẹ sáng mai nha.",
      [ASK("clinic.book_appointment", "doctor_id", "time")], CL),
    T("clinic_list", NOW, "Tuần này tôi có lịch khám nào không?", [C("clinic.list_appointments")], CL),
    T("clinic_slots", NOW, "Bác sĩ nhi Thu Trang còn trống những giờ nào?", [C("clinic.search_doctors", query="Thu Trang")], CL),
    T("clinic_out_of_scope", NOW, "Mua giúp anh hộp thuốc ho ở nhà thuốc gần nhất.", [], CL),
]
