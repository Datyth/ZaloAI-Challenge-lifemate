"""Tasks: ev_charging, expense_tracker, school_portal, apartment, sports_court, insurance, cloud_drive. Anchor Thursday 2026-11-12."""
from _common import ASK, C, CALENDAR, CONTACTS, REMINDERS, SMS, T

NOW = "2026-11-12T09:00:00+07:00"
NOON = "2026-11-12T12:15:00+07:00"

EV = {"vehicle": {"model": "VF 8", "battery_percent": 28, "connector": "ccs2", "range_km": 110}, "stations": [
    {"id": "evs_01", "name": "Trạm sạc Vincom Bà Triệu", "address": "191 Bà Triệu, Hai Bà Trưng, Hà Nội", "connectors": ["ccs2", "type2"], "power_kw": 60, "free_ports": 2, "price_per_kwh": 3858},
    {"id": "evs_02", "name": "Trạm sạc Aeon Long Biên", "address": "27 Cổ Linh, Long Biên, Hà Nội", "connectors": ["ccs2"], "power_kw": 150, "free_ports": 0, "price_per_kwh": 3858},
    {"id": "evs_03", "name": "Trạm sạc Times City", "address": "458 Minh Khai, Hai Bà Trưng, Hà Nội", "connectors": ["type2"], "power_kw": 11, "free_ports": 4, "price_per_kwh": 3500},
    {"id": "evs_04", "name": "Trạm sạc Royal City", "address": "72 Nguyễn Trãi, Thanh Xuân, Hà Nội", "connectors": ["ccs2", "chademo"], "power_kw": 120, "free_ports": 1, "price_per_kwh": 3858},
], "sessions": [{"id": "chg_01", "station_id": "evs_03", "start": "2026-11-13T22:00:00+07:00", "duration_min": 240, "target_percent": None, "status": "reserved"}]}
E = {"ev_charging": EV}
ER = {"ev_charging": EV, "reminders": REMINDERS}

EXP = {"categories": [
    {"id": "cat_01", "name": "Ăn uống", "budget": 4000000}, {"id": "cat_02", "name": "Đi lại", "budget": 1200000},
    {"id": "cat_03", "name": "Mua sắm", "budget": 2000000}, {"id": "cat_04", "name": "Hoá đơn", "budget": 3000000},
], "expenses": [
    {"id": "exp_01", "category_id": "cat_01", "amount": 85000, "date": "2026-11-11", "note": "bún chả"},
    {"id": "exp_02", "category_id": "cat_02", "amount": 250000, "date": "2026-11-10", "note": "đổ xăng"},
    {"id": "exp_03", "category_id": "cat_03", "amount": 1850000, "date": "2026-11-08", "note": "giày chạy bộ"},
    {"id": "exp_04", "category_id": "cat_01", "amount": 120000, "date": "2026-11-12", "note": "cà phê với khách"},
]}
X = {"expense_tracker": EXP}

SCHOOL = {"children": [
    {"id": "ch_01", "name": "Nguyễn Gia Bảo", "nickname": "Bin", "class": "5A2", "school": "Tiểu học Kim Đồng", "teacher": "Cô Hạnh"},
    {"id": "ch_02", "name": "Nguyễn Minh Anh", "nickname": "Na", "class": "2B1", "school": "Tiểu học Kim Đồng", "teacher": "Cô Phương"},
], "grades": [
    {"child_id": "ch_01", "subject": "Toán", "score": 9.5, "date": "2026-11-06"}, {"child_id": "ch_01", "subject": "Tiếng Việt", "score": 7.0, "date": "2026-11-09"},
    {"child_id": "ch_02", "subject": "Toán", "score": 10, "date": "2026-11-10"},
], "notices": [{"id": "nt_01", "class": "all", "title": "Họp phụ huynh cuối kỳ", "date": "2026-11-21"}, {"id": "nt_02", "class": "5A2", "title": "Nộp tiền quỹ lớp 200.000đ", "date": "2026-11-15"}],
    "leave_requests": [], "messages": []}
SCH = {"school_portal": SCHOOL}
SCHC = {"school_portal": SCHOOL, "calendar": CALENDAR}

APT = {"apartment": {"building": "Toà S2", "unit": "1205"}, "fees": [
    {"id": "fee_01", "type": "management", "month": "2026-11", "amount": 780000, "paid": False},
    {"id": "fee_02", "type": "parking", "month": "2026-11", "amount": 1200000, "paid": False},
    {"id": "fee_03", "type": "water", "month": "2026-10", "amount": 215000, "paid": True},
    {"id": "fee_04", "type": "water", "month": "2026-11", "amount": 236000, "paid": False},
], "requests": [], "guests": []}
AP = {"apartment": APT}

COURTS = {"courts": [
    {"id": "ct_01", "name": "Sân cầu lông Ánh Dương", "sport": "badminton", "address": "Ngõ 34 Hoàng Cầu, Đống Đa", "price_per_hour": 120000,
     "slots": {"2026-11-12": ["18:00", "19:00", "20:00"], "2026-11-13": ["06:00", "19:00"], "2026-11-14": ["07:00", "08:00", "09:00"]}},
    {"id": "ct_02", "name": "Sân bóng mini Hồ Tây", "sport": "football", "address": "614 Lạc Long Quân, Tây Hồ", "price_per_hour": 600000,
     "slots": {"2026-11-12": ["19:00", "20:00"], "2026-11-14": ["16:00", "17:00", "18:00"]}},
    {"id": "ct_03", "name": "Pickleball Cầu Giấy", "sport": "pickleball", "address": "Công viên Cầu Giấy", "price_per_hour": 200000,
     "slots": {"2026-11-13": ["17:00", "18:00"]}},
], "bookings": [{"id": "cb_01", "court_id": "ct_01", "date": "2026-11-13", "start_time": "06:00", "hours": 1, "total": 120000, "status": "booked"}]}
CT = {"sports_court": COURTS}
CTS = {"sports_court": COURTS, "contacts": CONTACTS, "sms": SMS}

INS = {"policies": [
    {"id": "pol_01", "type": "car", "insurer": "Bảo hiểm An Tâm", "plate": "30H-123.45", "premium": 6500000, "end_date": "2026-11-30", "status": "active"},
    {"id": "pol_02", "type": "health", "insurer": "Sức Khoẻ Vàng", "premium": 9800000, "end_date": "2027-03-01", "status": "active"},
    {"id": "pol_03", "type": "motorbike", "insurer": "Bảo hiểm An Tâm", "plate": "29B1-678.90", "premium": 66000, "end_date": "2026-10-20", "status": "expired"},
], "claims": []}
IN = {"insurance": INS}

DRIVE = {"files": [
    {"id": "f_01", "name": "Báo cáo quý 3.xlsx", "folder": "Công việc", "size_mb": 2.4, "shared_with": [], "trashed": False},
    {"id": "f_02", "name": "Hợp đồng thuê nhà.pdf", "folder": "Giấy tờ", "size_mb": 1.1, "shared_with": [], "trashed": False},
    {"id": "f_03", "name": "Ảnh du lịch Đà Nẵng.zip", "folder": "Ảnh", "size_mb": 830, "shared_with": [], "trashed": False},
    {"id": "f_04", "name": "Slide họp khách hàng.pptx", "folder": "Công việc", "size_mb": 15, "shared_with": [], "trashed": False},
    {"id": "f_05", "name": "Báo cáo quý 3 - bản nháp.xlsx", "folder": "Tải về", "size_mb": 2.1, "shared_with": [], "trashed": False},
]}
D = {"cloud_drive": DRIVE}
DC = {"cloud_drive": DRIVE, "contacts": CONTACTS}

TASKS = [
    # ev_charging
    T("ev_low_battery", NOW, "Pin xe còn dưới ba mươi phần trăm thì đặt sạc ở trạm Vincom Bà Triệu lúc mười giờ, sạc một tiếng nhé.",
      [C("ev_charging.get_vehicle"), C("ev_charging.find_stations", query="Vincom Bà Triệu"),
       C("ev_charging.reserve", station_id="evs_01", start="2026-11-12T10:00:00+07:00", duration_min=60)], E),
    T("ev_fast", NOW, "Tìm trạm sạc nhanh trên một trăm kW chuẩn CCS2 còn chỗ, đặt lúc hai giờ chiều nay bốn mươi lăm phút, sạc tới tám mươi phần trăm.",
      [C("ev_charging.find_stations", connector="ccs2", min_power_kw=100),
       C("ev_charging.reserve", station_id="evs_04", start="2026-11-12T14:00:00+07:00", duration_min=45, target_percent=80)], E),
    T("ev_cancel", NOW, "Tối mai không sạc ở Times City nữa, hủy giúp anh.", [C("ev_charging.list_sessions"), C("ev_charging.cancel_session", session_id="chg_01")], E),
    T("ev_missing_duration", NOW, "Đặt lịch sạc ở Royal City lúc năm giờ chiều.", [C("ev_charging.find_stations", query="Royal City"), ASK("ev_charging.reserve", "duration_min")], E),
    T("ev_battery", NOW, "Xe còn bao nhiêu phần trăm pin?", [C("ev_charging.get_vehicle")], E),
    T("ev_full_station", NOW, "Đặt sạc ở Aeon Long Biên lúc ba giờ chiều, ba mươi phút.", [C("ev_charging.find_stations", query="Aeon Long Biên")], E),
    T("ev_reserve_remind", NOW, "Đặt sạc ở Times City tám giờ tối nay hai tiếng, nhắc anh trước mười lăm phút.",
      [C("ev_charging.find_stations", query="Times City"), C("ev_charging.reserve", station_id="evs_03", start="2026-11-12T20:00:00+07:00", duration_min=120),
       C("reminders.create", content="đi sạc xe ở Times City", time="2026-11-12T19:45:00+07:00")], ER),
    # expense_tracker
    T("exp_add_food", NOON, "Ghi chi ăn trưa bảy mươi lăm nghìn.", [C("expense_tracker.list_categories"), C("expense_tracker.add_expense", amount=75000, category_id="cat_01", note="ăn trưa")], X),
    T("exp_add_yesterday", NOON, "Hôm qua đi grab hết một trăm hai mươi nghìn, ghi vào mục đi lại.",
      [C("expense_tracker.list_categories"), C("expense_tracker.add_expense", amount=120000, category_id="cat_02", date="2026-11-11", note="đi grab")], X),
    T("exp_budget", NOON, "Tăng ngân sách mua sắm tháng này lên ba triệu.", [C("expense_tracker.list_categories"), C("expense_tracker.set_budget", category_id="cat_03", budget=3000000)], X),
    T("exp_delete", NOON, "Khoản cà phê với khách sáng nay ghi nhầm, xóa đi.",
      [C("expense_tracker.list_categories"), C("expense_tracker.list_expenses", category_id="cat_01", date_from="2026-11-12", date_to="2026-11-12"),
       C("expense_tracker.delete_expense", expense_id="exp_04")], X),
    T("exp_missing_amount", NOON, "Ghi giúp chị khoản tiền điện tháng này.", [C("expense_tracker.list_categories"), ASK("expense_tracker.add_expense", "amount")], X),
    T("exp_how_much", NOON, "Tháng này ăn uống tiêu hết bao nhiêu rồi?", [C("expense_tracker.list_categories")], X),
    T("exp_conditional", NOON, "Nếu mua sắm tháng này vượt một triệu rưỡi thì giảm ngân sách mua sắm xuống còn một triệu tám.",
      [C("expense_tracker.list_categories"), C("expense_tracker.set_budget", category_id="cat_03", budget=1800000)], X),
    # school_portal
    T("school_leave", NOW, "Xin cho bé Na nghỉ học ngày mai vì bé bị sốt.",
      [C("school_portal.list_children"), C("school_portal.request_leave", child_id="ch_02", date="2026-11-13", reason="bị sốt")], SCH),
    T("school_leave_days", NOW, "Xin phép cho Bin nghỉ hai ngày từ thứ hai tuần sau, nhà có việc về quê.",
      [C("school_portal.list_children"), C("school_portal.request_leave", child_id="ch_01", date="2026-11-16", days=2, reason="nhà có việc về quê")], SCH),
    T("school_grades", NOW, "Bin dạo này điểm toán thế nào?", [C("school_portal.list_children"), C("school_portal.get_grades", child_id="ch_01", subject="Toán")], SCH),
    T("school_msg", NOW, "Nhắn cô chủ nhiệm của Na là chiều nay bố đón muộn mười lăm phút.",
      [C("school_portal.list_children"), C("school_portal.message_teacher", child_id="ch_02", content="Chiều nay bố đón muộn mười lăm phút")], SCH),
    T("school_missing_reason", NOW, "Xin cho Bin nghỉ học thứ sáu.", [C("school_portal.list_children"), ASK("school_portal.request_leave", "reason")], SCH),
    T("school_notice_cal", NOW, "Trường có lịch họp phụ huynh không, có thì thêm vào lịch của anh lúc tám giờ sáng hôm đó.",
      [C("school_portal.get_notices"), C("calendar.create_event", title="Họp phụ huynh cuối kỳ", start="2026-11-21T08:00:00+07:00")], SCHC),
    # apartment
    T("apt_pay_unpaid", NOW, "Đóng hết các khoản phí chung cư chưa đóng của tháng mười một.",
      [C("apartment.list_fees", paid=False), C("apartment.pay_fee", fee_id="fee_01"), C("apartment.pay_fee", fee_id="fee_02"), C("apartment.pay_fee", fee_id="fee_04")], AP),
    T("apt_pay_parking", NOW, "Trả phí gửi xe tháng này giúp tôi.", [C("apartment.list_fees", paid=False), C("apartment.pay_fee", fee_id="fee_02")], AP),
    T("apt_leak", NOW, "Báo ban quản lý là bồn rửa bát nhà mình bị rò nước, hẹn thợ lên lúc hai giờ chiều mai.",
      [C("apartment.create_request", category="plumbing", description="bồn rửa bát bị rò nước", preferred_time="2026-11-13T14:00:00+07:00")], AP),
    T("apt_elevator", NOW, "Thang máy số hai cứ kêu lạch cạch, báo ban quản lý giùm.", [C("apartment.create_request", category="elevator", description="thang máy số hai kêu lạch cạch")], AP),
    T("apt_guest", NOW, "Đăng ký khách là anh Phong tới nhà lúc bảy giờ tối nay, đi ô tô biển 30A-456.78.",
      [C("apartment.register_guest", name="anh Phong", arrival="2026-11-12T19:00:00+07:00", vehicle_plate="30A-456.78")], AP),
    T("apt_guest_missing", NOW, "Đăng ký khách đến chơi tối nay nhé.", [ASK("apartment.register_guest", "name", "arrival")], AP),
    T("apt_how_much", NOW, "Tháng này phí quản lý bao nhiêu?", [C("apartment.list_fees")], AP),
    # sports_court
    T("court_badminton_tonight", NOW, "Đặt sân cầu lông bảy giờ tối nay hai tiếng.",
      [C("sports_court.search_courts", sport="badminton", date="2026-11-12"), C("sports_court.book_court", court_id="ct_01", date="2026-11-12", start_time="19:00", hours=2)], CT),
    T("court_football_sat", NOW, "Thứ bảy này đặt sân bóng mini năm giờ chiều một tiếng rồi nhắn anh Tuấn hàng xóm ra đá.",
      [C("sports_court.search_courts", sport="football", date="2026-11-14"), C("sports_court.book_court", court_id="ct_02", date="2026-11-14", start_time="17:00", hours=1),
       C("contacts.search", query="Anh Tuấn"), C("sms.send", contact_id="c_07", content="Thứ bảy này 5 giờ chiều ra sân bóng mini Hồ Tây đá bóng nhé")], CTS),
    T("court_cancel", NOW, "Hủy sân cầu lông sáng mai đi.", [C("sports_court.list_bookings"), C("sports_court.cancel_booking", booking_id="cb_01")], CT),
    T("court_missing_time", NOW, "Đặt sân pickleball ngày mai giúp mình.", [C("sports_court.search_courts", sport="pickleball", date="2026-11-13"), ASK("sports_court.book_court", "start_time", "hours")], CT),
    T("court_full", NOW, "Đặt sân bóng tối nay sáu giờ.", [C("sports_court.search_courts", sport="football", date="2026-11-12")], CT),
    T("court_out_of_scope", NOW, "Thuê giúp anh một huấn luyện viên tennis.", [], CT),
    # insurance
    T("ins_renew_car", NOW, "Bảo hiểm ô tô sắp hết hạn, gia hạn thêm một năm nhé.", [C("insurance.list_policies", type="car"), C("insurance.renew_policy", policy_id="pol_01")], IN),
    T("ins_renew_bike_6", NOW, "Gia hạn bảo hiểm xe máy sáu tháng.", [C("insurance.list_policies", type="motorbike"), C("insurance.renew_policy", policy_id="pol_03", months=6)], IN),
    T("ins_claim", NOW, "Hôm qua anh đi khám hết hai triệu ba, làm hồ sơ bồi thường bảo hiểm sức khỏe giúp anh, lý do khám viêm họng.",
      [C("insurance.list_policies", type="health"),
       C("insurance.file_claim", policy_id="pol_02", incident_date="2026-11-11", amount=2300000, description="khám viêm họng")], IN),
    T("ins_claim_missing", NOW, "Xe ô tô bị quẹt hôm thứ hai, gửi yêu cầu bồi thường đi.", [C("insurance.list_policies", type="car"), ASK("insurance.file_claim", "amount")], IN),
    T("ins_when", NOW, "Bảo hiểm sức khỏe của tôi tới khi nào hết hạn?", [C("insurance.list_policies", type="health")], IN),
    # cloud_drive
    T("drive_share_contact", NOW, "Chia sẻ file báo cáo quý ba cho anh Khoa, cho sửa luôn.",
      [C("cloud_drive.search_files", query="Báo cáo quý 3"), C("contacts.search", query="Anh Khoa"),
       C("cloud_drive.share_file", file_id="f_01", contact_id="c_02", permission="edit")], DC),
    T("drive_share_email", NOW, "Gửi quyền xem hợp đồng thuê nhà cho email chủ nhà là chu.nha.88@mail.vn.",
      [C("cloud_drive.search_files", query="Hợp đồng thuê nhà"), C("cloud_drive.share_file", file_id="f_02", email="chu.nha.88@mail.vn", permission="view")], D),
    T("drive_trash_draft", NOW, "Xóa cái bản nháp báo cáo quý ba trong thư mục tải về đi.",
      [C("cloud_drive.search_files", query="bản nháp"), C("cloud_drive.trash_file", file_id="f_05")], D),
    T("drive_move", NOW, "Chuyển slide họp khách hàng sang thư mục Khách hàng ABC.",
      [C("cloud_drive.search_files", query="Slide họp khách hàng"), C("cloud_drive.move_file", file_id="f_04", folder="Khách hàng ABC")], D),
    T("drive_share_missing", NOW, "Chia sẻ file ảnh du lịch Đà Nẵng nhé.", [C("cloud_drive.search_files", query="Đà Nẵng"), ASK("cloud_drive.share_file", "contact_id")], D),
    T("drive_list_work", NOW, "Trong thư mục công việc có những file gì?", [C("cloud_drive.search_files", folder="Công việc")], D),
]
