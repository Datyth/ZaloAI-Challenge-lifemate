"""Catalog entries of the synthetic apps (same format as <split>_tools_catalog.json), written with small helpers.

The synthetic apps are new everyday Vietnamese services, written from scratch for training only. None copies a TRAIN app, and the
domains of the PUBLIC-only apps are left out on purpose, so that public stays an honest test of transfer to unseen apps.
Conventions follow the TRAIN catalog: Vietnamese descriptions, "ID ..., lấy từ <tool>" with x-ref, fuzzy free text, ISO formats."""
from __future__ import annotations

ISO = "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"
DATE = "YYYY-MM-DD, ví dụ 2026-11-29"
HHMM = "HH:MM, ví dụ 18:30"
INT = "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def s(desc=None, enum=None, fuzzy=False, ref=None, fmt=None, min_len=None):
    p = {"type": "string"}
    if desc:
        p["description"] = desc
    if enum:
        p["enum"] = enum
    if fuzzy:
        p["x-match"] = "fuzzy"
    if ref:
        p["x-ref"] = ref
    if fmt:
        p["format"] = fmt
    if min_len:
        p["minLength"] = min_len
    return p


def i(desc=INT, lo=None, hi=None, enum=None):
    p = {"type": "integer", "description": desc}
    if lo is not None:
        p["minimum"] = lo
    if hi is not None:
        p["maximum"] = hi
    if enum:
        p["enum"] = enum
    return p


def n(desc, lo=None, hi=None):
    p = {"type": "number", "description": desc}
    if lo is not None:
        p["minimum"] = lo
    if hi is not None:
        p["maximum"] = hi
    return p


def b(desc):
    return {"type": "boolean", "description": desc}


def arr(items, desc=None):
    p = {"type": "array", "items": items}
    if desc:
        p["description"] = desc
    return p


def obj(props, required):
    return {"type": "object", "properties": props, "required": required}


def ref_id(what, tool, ref):
    return s(f"ID {what}, lấy từ {tool}", ref=ref)


def tool(app, name, kind, desc, props=None, required=(), returns="", one_of=None):
    params = {"type": "object", "properties": props or {}, "required": list(required)}
    if one_of:
        params["x-one-of"] = one_of
    return {"name": f"{app}.{name}", "kind": kind, "description": desc, "parameters": params, "returns": returns, "app": app}


def app(id_, name_vi, group, desc, tools):
    return {"id": id_, "name_vi": name_vi, "group": group, "description": desc, "tools": tools}


A = "hotel_booking"
HOTEL = app(A, "Đặt phòng khách sạn", "travel", "Tìm khách sạn, đặt phòng, đổi ngày hoặc huỷ đặt phòng", [
    tool(A, "search_hotels", "read", "Tìm khách sạn theo thành phố, tên hoặc giá tối đa một đêm; không có tham số thì trả mọi khách sạn",
         {"city": s("Thành phố, ví dụ Đà Lạt"), "query": s("Tên hoặc khu vực khách sạn"), "max_price": i("Giá tối đa một đêm (đồng). " + INT, lo=0)},
         returns="hotels[]"),
    tool(A, "list_bookings", "read", "Liệt kê các đặt phòng của người dùng", {"status": s(enum=["confirmed", "cancelled"])}, returns="bookings[]"),
    tool(A, "book", "write", "Đặt phòng; bắt buộc khách sạn, loại phòng, ngày nhận, ngày trả và số khách",
         {"hotel_id": ref_id("khách sạn", "hotel_booking.search_hotels", "hotel_booking.hotels"),
          "room_type": s("standard = phòng thường, deluxe = phòng cao cấp, family = phòng gia đình", enum=["standard", "deluxe", "family"]),
          "check_in": s(DATE, fmt="date"), "check_out": s(DATE, fmt="date"), "guests": i("Số khách. " + INT, lo=1, hi=8),
          "note": s("Yêu cầu thêm, chỉ khi người dùng nói", fuzzy=True)},
         ["hotel_id", "room_type", "check_in", "check_out", "guests"]),
    tool(A, "change_dates", "write", "Đổi ngày nhận/trả phòng của một đặt phòng",
         {"booking_id": ref_id("đặt phòng", "hotel_booking.list_bookings", "hotel_booking.bookings"), "check_in": s(DATE, fmt="date"), "check_out": s(DATE, fmt="date")},
         ["booking_id", "check_in", "check_out"]),
    tool(A, "cancel_booking", "write", "Huỷ một đặt phòng", {"booking_id": ref_id("đặt phòng", "hotel_booking.list_bookings", "hotel_booking.bookings")}, ["booking_id"]),
])

A = "clinic"
CLINIC = app(A, "Đặt lịch khám", "health", "Tìm bác sĩ, đặt, đổi hoặc huỷ lịch khám bệnh", [
    tool(A, "search_doctors", "read", "Tìm bác sĩ theo chuyên khoa hoặc tên; kết quả có các khung giờ còn trống (available_slots)",
         {"specialty": s("general = đa khoa, pediatrics = nhi, dentistry = răng hàm mặt, dermatology = da liễu, cardiology = tim mạch, ent = tai mũi họng, ophthalmology = mắt",
                         enum=["general", "pediatrics", "dentistry", "dermatology", "cardiology", "ent", "ophthalmology"]),
          "query": s("Tên bác sĩ hoặc phòng khám")}, returns="doctors[]"),
    tool(A, "list_appointments", "read", "Liệt kê lịch khám đã đặt", {}, returns="appointments[]"),
    tool(A, "book_appointment", "write", "Đặt lịch khám vào một khung giờ còn trống của bác sĩ",
         {"doctor_id": ref_id("bác sĩ", "clinic.search_doctors", "clinic.doctors"), "time": s(ISO, fmt="date-time"),
          "patient": s("Người được khám; bỏ trống = chính người dùng", fuzzy=True), "reason": s("Lý do khám, chỉ khi người dùng nói", fuzzy=True)},
         ["doctor_id", "time"]),
    tool(A, "reschedule", "write", "Đổi giờ một lịch khám sang khung giờ còn trống khác",
         {"appointment_id": ref_id("lịch khám", "clinic.list_appointments", "clinic.appointments"), "time": s(ISO, fmt="date-time")}, ["appointment_id", "time"]),
    tool(A, "cancel_appointment", "write", "Huỷ một lịch khám", {"appointment_id": ref_id("lịch khám", "clinic.list_appointments", "clinic.appointments")}, ["appointment_id"]),
])

A = "pet_care"
PET = app(A, "Chăm sóc thú cưng", "personal", "Theo dõi thú cưng: ghi lại cho ăn, dắt đi dạo, tiêm phòng; lên lịch chăm sóc", [
    tool(A, "list_pets", "read", "Danh sách thú cưng", {}, returns="pets[]"),
    tool(A, "get_logs", "read", "Nhật ký chăm sóc, lọc theo thú cưng và loại",
         {"pet_id": ref_id("thú cưng", "pet_care.list_pets", "pet_care.pets"), "type": s(enum=["feeding", "walk", "vaccine", "vet_visit", "medicine", "bath"])}, returns="logs[]"),
    tool(A, "log_activity", "write", "Ghi một hoạt động chăm sóc vừa làm",
         {"pet_id": ref_id("thú cưng", "pet_care.list_pets", "pet_care.pets"),
          "type": s("feeding = cho ăn, walk = dắt đi dạo, vaccine = tiêm phòng, vet_visit = khám thú y, medicine = cho uống thuốc, bath = tắm", enum=["feeding", "walk", "vaccine", "vet_visit", "medicine", "bath"]),
          "amount": n("Lượng (gam thức ăn, phút đi dạo...), chỉ khi người dùng nói", lo=0), "note": s(fuzzy=True)}, ["pet_id", "type"]),
    tool(A, "schedule_care", "write", "Lên lịch một việc chăm sóc",
         {"pet_id": ref_id("thú cưng", "pet_care.list_pets", "pet_care.pets"), "type": s(enum=["feeding", "walk", "vaccine", "vet_visit", "medicine", "bath"]),
          "time": s(ISO, fmt="date-time"), "repeat": s("Lặp lại; bỏ trống = một lần", enum=["none", "daily", "weekly", "monthly"])}, ["pet_id", "type", "time"]),
    tool(A, "update_weight", "write", "Cập nhật cân nặng thú cưng", {"pet_id": ref_id("thú cưng", "pet_care.list_pets", "pet_care.pets"), "weight_kg": n("Cân nặng (kg)", lo=0)}, ["pet_id", "weight_kg"]),
])

A = "laundry"
LAUNDRY = app(A, "Giặt là", "home", "Đặt dịch vụ giặt sấy, giặt hấp lấy tận nơi; xem và huỷ đơn", [
    tool(A, "list_services", "read", "Bảng giá các dịch vụ giặt", {}, returns="services[]"),
    tool(A, "list_orders", "read", "Các đơn giặt", {"status": s(enum=["scheduled", "washing", "delivered", "cancelled"])}, returns="orders[]"),
    tool(A, "create_order", "write", "Tạo đơn giặt; items là các dịch vụ và số lượng (kg hoặc chiếc theo đơn vị của dịch vụ)",
         {"items": arr(obj({"service_id": s("ID dịch vụ, lấy từ laundry.list_services"), "quantity": n("Số lượng theo đơn vị của dịch vụ", lo=1)}, ["service_id", "quantity"])),
          "pickup_time": s(ISO, fmt="date-time"), "address": s("Địa chỉ lấy đồ; bỏ trống = địa chỉ mặc định của tài khoản", fuzzy=True),
          "express": b("Giặt nhanh trong ngày, chỉ khi người dùng yêu cầu")}, ["items", "pickup_time"]),
    tool(A, "cancel_order", "write", "Huỷ một đơn giặt chưa lấy đồ", {"order_id": ref_id("đơn giặt", "laundry.list_orders", "laundry.orders")}, ["order_id"]),
])

A = "salon"
SALON = app(A, "Salon tóc & spa", "personal", "Tìm salon, xem dịch vụ, đặt lịch cắt tóc, gội, làm móng, spa", [
    tool(A, "search_salons", "read", "Tìm salon theo tên hoặc khu vực; không có query thì trả mọi salon", {"query": s("Tên hoặc khu vực")}, returns="salons[]"),
    tool(A, "get_salon", "read", "Chi tiết một salon: dịch vụ (services) và thợ (stylists)", {"salon_id": ref_id("salon", "salon.search_salons", "salon.salons")}, ["salon_id"], returns="salon"),
    tool(A, "book", "write", "Đặt lịch tại salon cho một hoặc nhiều dịch vụ",
         {"salon_id": ref_id("salon", "salon.search_salons", "salon.salons"), "service_ids": arr({"type": "string"}, "ID các dịch vụ, lấy từ salon.get_salon"),
          "time": s(ISO, fmt="date-time"), "stylist_id": s("ID thợ, lấy từ salon.get_salon; chỉ khi người dùng chọn thợ")}, ["salon_id", "service_ids", "time"]),
    tool(A, "list_bookings", "read", "Các lịch đã đặt", {}, returns="bookings[]"),
    tool(A, "cancel_booking", "write", "Huỷ một lịch đã đặt", {"booking_id": ref_id("lịch hẹn", "salon.list_bookings", "salon.bookings")}, ["booking_id"]),
])

A = "ev_charging"
EV = app(A, "Sạc xe điện", "transport", "Xem pin xe, tìm trạm sạc, đặt lịch sạc", [
    tool(A, "get_vehicle", "read", "Thông tin xe và phần trăm pin hiện tại", {}, returns="vehicle"),
    tool(A, "find_stations", "read", "Tìm trạm sạc theo chuẩn cổng, công suất tối thiểu hoặc tên/khu vực",
         {"connector": s(enum=["ccs2", "type2", "chademo"]), "min_power_kw": i("Công suất tối thiểu (kW). " + INT, lo=0), "query": s("Tên hoặc khu vực")}, returns="stations[]"),
    tool(A, "reserve", "write", "Đặt một lượt sạc tại trạm còn cổng trống",
         {"station_id": ref_id("trạm sạc", "ev_charging.find_stations", "ev_charging.stations"), "start": s(ISO, fmt="date-time"),
          "duration_min": i("Thời lượng (phút). " + INT, lo=15, hi=240), "target_percent": i("Sạc tới bao nhiêu phần trăm, chỉ khi người dùng nói. " + INT, lo=50, hi=100)},
         ["station_id", "start", "duration_min"]),
    tool(A, "list_sessions", "read", "Các lượt sạc đã đặt", {}, returns="sessions[]"),
    tool(A, "cancel_session", "write", "Huỷ một lượt sạc", {"session_id": ref_id("lượt sạc", "ev_charging.list_sessions", "ev_charging.sessions")}, ["session_id"]),
])

A = "wifi_router"
WIFI = app(A, "Wi-Fi gia đình", "home", "Quản lý mạng Wi-Fi: thiết bị kết nối, chặn thiết bị, mạng khách, giờ tắt mạng", [
    tool(A, "get_status", "read", "Trạng thái mạng và các thiết bị đang kết nối", {}, returns="network, devices[]"),
    tool(A, "block_device", "write", "Chặn hoặc bỏ chặn một thiết bị",
         {"device_id": ref_id("thiết bị", "wifi_router.get_status", "wifi_router.devices"), "blocked": b("true = chặn, false = bỏ chặn")}, ["device_id", "blocked"]),
    tool(A, "set_guest_network", "write", "Bật hoặc tắt mạng khách",
         {"enabled": b("true = bật, false = tắt"), "password": s("Mật khẩu mạng khách, ít nhất 8 ký tự; chỉ khi người dùng đọc", min_len=8)}, ["enabled"]),
    tool(A, "set_schedule", "write", "Hẹn giờ ngắt mạng của một thiết bị (ví dụ giờ đi ngủ của con)",
         {"device_id": ref_id("thiết bị", "wifi_router.get_status", "wifi_router.devices"), "off_from": s(HHMM, fmt="time"), "off_to": s(HHMM, fmt="time"),
          "days": arr({"type": "string", "enum": DAYS}, "Các ngày áp dụng; bỏ trống = mọi ngày")}, ["device_id", "off_from", "off_to"]),
    tool(A, "change_password", "write", "Đổi mật khẩu Wi-Fi chính", {"new_password": s("Mật khẩu mới, ít nhất 8 ký tự", min_len=8)}, ["new_password"]),
])

A = "expense_tracker"
EXPENSE = app(A, "Sổ chi tiêu", "finance", "Ghi chép chi tiêu theo danh mục và ngân sách tháng", [
    tool(A, "list_categories", "read", "Danh mục chi tiêu, ngân sách tháng (budget) và số đã chi tháng này (spent)", {}, returns="categories[]"),
    tool(A, "list_expenses", "read", "Các khoản chi, lọc theo danh mục và khoảng ngày",
         {"category_id": ref_id("danh mục", "expense_tracker.list_categories", "expense_tracker.categories"), "date_from": s(DATE), "date_to": s(DATE)}, returns="expenses[]"),
    tool(A, "add_expense", "write", "Ghi một khoản chi",
         {"amount": i("Số tiền (đồng). " + INT, lo=1), "category_id": ref_id("danh mục", "expense_tracker.list_categories", "expense_tracker.categories"),
          "date": s(DATE + "; bỏ trống = hôm nay", fmt="date"), "note": s(fuzzy=True)}, ["amount", "category_id"]),
    tool(A, "set_budget", "write", "Đặt ngân sách tháng của một danh mục",
         {"category_id": ref_id("danh mục", "expense_tracker.list_categories", "expense_tracker.categories"), "budget": i("Ngân sách (đồng). " + INT, lo=0)}, ["category_id", "budget"]),
    tool(A, "delete_expense", "write", "Xoá một khoản chi ghi nhầm", {"expense_id": ref_id("khoản chi", "expense_tracker.list_expenses", "expense_tracker.expenses")}, ["expense_id"]),
])

A = "school_portal"
SCHOOL = app(A, "Sổ liên lạc điện tử", "family", "Xem điểm, thông báo của con; xin nghỉ học; nhắn giáo viên chủ nhiệm", [
    tool(A, "list_children", "read", "Các con của người dùng (tên, lớp, trường)", {}, returns="children[]"),
    tool(A, "get_grades", "read", "Điểm của một học sinh, lọc theo môn",
         {"child_id": ref_id("học sinh", "school_portal.list_children", "school_portal.children"), "subject": s("Môn học, ví dụ Toán")}, ["child_id"], returns="grades[]"),
    tool(A, "get_notices", "read", "Thông báo của trường/lớp", {"child_id": ref_id("học sinh", "school_portal.list_children", "school_portal.children")}, returns="notices[]"),
    tool(A, "request_leave", "write", "Gửi đơn xin nghỉ học",
         {"child_id": ref_id("học sinh", "school_portal.list_children", "school_portal.children"), "date": s(DATE, fmt="date"),
          "days": i("Số ngày nghỉ; bỏ trống = 1. " + INT, lo=1, hi=5), "reason": s("Lý do", fuzzy=True)}, ["child_id", "date", "reason"]),
    tool(A, "message_teacher", "write", "Nhắn giáo viên chủ nhiệm của một học sinh",
         {"child_id": ref_id("học sinh", "school_portal.list_children", "school_portal.children"), "content": s(fuzzy=True)}, ["child_id", "content"]),
])

A = "apartment"
APARTMENT = app(A, "Cư dân chung cư", "home", "Phí dịch vụ chung cư, báo sửa chữa, đăng ký khách", [
    tool(A, "list_fees", "read", "Các khoản phí (quản lý, gửi xe, nước, điện) theo tháng", {"paid": b("true = đã đóng, false = chưa đóng")}, returns="fees[]"),
    tool(A, "pay_fee", "write", "Đóng một khoản phí", {"fee_id": ref_id("khoản phí", "apartment.list_fees", "apartment.fees")}, ["fee_id"]),
    tool(A, "create_request", "write", "Gửi yêu cầu sửa chữa / hỗ trợ cho ban quản lý",
         {"category": s("plumbing = điện nước rò rỉ, ống nước; electrical = điện; cleaning = vệ sinh; security = an ninh; elevator = thang máy; other = khác",
                        enum=["plumbing", "electrical", "cleaning", "security", "elevator", "other"]),
          "description": s(fuzzy=True), "preferred_time": s(ISO + "; chỉ khi người dùng hẹn giờ", fmt="date-time")}, ["category", "description"]),
    tool(A, "list_requests", "read", "Các yêu cầu đã gửi", {}, returns="requests[]"),
    tool(A, "register_guest", "write", "Đăng ký khách đến căn hộ",
         {"name": s("Tên khách", fuzzy=True), "arrival": s(ISO, fmt="date-time"), "vehicle_plate": s("Biển số xe, chỉ khi người dùng đọc")}, ["name", "arrival"]),
])

A = "sports_court"
COURT = app(A, "Đặt sân thể thao", "personal", "Tìm và đặt sân cầu lông, bóng đá mini, tennis, pickleball", [
    tool(A, "search_courts", "read", "Tìm sân theo môn và ngày; kết quả có các giờ bắt đầu còn trống (free_slots) của ngày đó",
         {"sport": s("badminton = cầu lông, football = bóng đá mini, tennis, pickleball", enum=["badminton", "football", "tennis", "pickleball"]), "date": s(DATE, fmt="date")},
         ["sport", "date"], returns="courts[]"),
    tool(A, "book_court", "write", "Đặt sân từ một giờ bắt đầu còn trống, theo số giờ",
         {"court_id": ref_id("sân", "sports_court.search_courts", "sports_court.courts"), "date": s(DATE, fmt="date"), "start_time": s(HHMM, fmt="time"),
          "hours": i("Số giờ. " + INT, lo=1, hi=3)}, ["court_id", "date", "start_time", "hours"]),
    tool(A, "list_bookings", "read", "Các lượt đặt sân", {}, returns="bookings[]"),
    tool(A, "cancel_booking", "write", "Huỷ một lượt đặt sân", {"booking_id": ref_id("lượt đặt", "sports_court.list_bookings", "sports_court.bookings")}, ["booking_id"]),
])

A = "insurance"
INSURANCE = app(A, "Bảo hiểm", "finance", "Hợp đồng bảo hiểm: xem, gia hạn, yêu cầu bồi thường", [
    tool(A, "list_policies", "read", "Các hợp đồng bảo hiểm, lọc theo loại",
         {"type": s("health = sức khoẻ, car = ô tô, motorbike = xe máy, home = nhà, travel = du lịch", enum=["health", "car", "motorbike", "home", "travel"])}, returns="policies[]"),
    tool(A, "renew_policy", "write", "Gia hạn một hợp đồng",
         {"policy_id": ref_id("hợp đồng", "insurance.list_policies", "insurance.policies"), "months": i("Số tháng gia hạn; bỏ trống = 12", enum=[6, 12])}, ["policy_id"]),
    tool(A, "file_claim", "write", "Gửi yêu cầu bồi thường",
         {"policy_id": ref_id("hợp đồng", "insurance.list_policies", "insurance.policies"), "incident_date": s(DATE, fmt="date"),
          "amount": i("Số tiền yêu cầu (đồng). " + INT, lo=1), "description": s(fuzzy=True)}, ["policy_id", "incident_date", "amount", "description"]),
    tool(A, "list_claims", "read", "Các yêu cầu bồi thường đã gửi", {}, returns="claims[]"),
])

A = "cloud_drive"
DRIVE = app(A, "Lưu trữ đám mây", "work", "Tìm, chia sẻ, chuyển thư mục, xoá tệp trên ổ đám mây", [
    tool(A, "search_files", "read", "Tìm tệp theo tên hoặc thư mục; không có tham số thì trả mọi tệp", {"query": s("Tên tệp"), "folder": s("Tên thư mục")}, returns="files[]"),
    tool(A, "share_file", "write", "Chia sẻ một tệp cho một liên hệ hoặc một email",
         {"file_id": ref_id("tệp", "cloud_drive.search_files", "cloud_drive.files"), "contact_id": s("ID liên hệ, lấy từ contacts.search", ref="contacts.contacts"),
          "email": s("Email, giữ nguyên chuỗi người dùng đọc"), "permission": s("view = chỉ xem, edit = được sửa; bỏ trống = view", enum=["view", "edit"])},
         ["file_id"], one_of=[["contact_id"], ["email"]]),
    tool(A, "move_file", "write", "Chuyển tệp sang thư mục khác (tạo thư mục nếu chưa có)",
         {"file_id": ref_id("tệp", "cloud_drive.search_files", "cloud_drive.files"), "folder": s("Tên thư mục đích")}, ["file_id", "folder"]),
    tool(A, "trash_file", "write", "Chuyển tệp vào thùng rác", {"file_id": ref_id("tệp", "cloud_drive.search_files", "cloud_drive.files")}, ["file_id"]),
])

A = "smart_garden"
GARDEN = app(A, "Vườn thông minh", "home", "Tưới cây tự động: độ ẩm từng khu, tưới ngay, lịch tưới", [
    tool(A, "get_status", "read", "Các khu vườn (zones) với độ ẩm đất (%) và lịch tưới (schedules)", {}, returns="zones[], schedules[]"),
    tool(A, "water_now", "write", "Tưới ngay một khu trong số phút",
         {"zone_id": ref_id("khu vườn", "smart_garden.get_status", "smart_garden.zones"), "duration_min": i("Số phút. " + INT, lo=1, hi=60)}, ["zone_id", "duration_min"]),
    tool(A, "set_schedule", "write", "Đặt lịch tưới hằng ngày hoặc theo thứ cho một khu",
         {"zone_id": ref_id("khu vườn", "smart_garden.get_status", "smart_garden.zones"), "time": s(HHMM, fmt="time"), "duration_min": i("Số phút. " + INT, lo=1, hi=60),
          "days": arr({"type": "string", "enum": DAYS}, "Các ngày tưới; bỏ trống = mọi ngày")}, ["zone_id", "time", "duration_min"]),
    tool(A, "pause_schedule", "write", "Tạm dừng một lịch tưới đến hết một ngày (ví dụ khi trời mưa)",
         {"schedule_id": ref_id("lịch tưới", "smart_garden.get_status", "smart_garden.schedules"), "until": s(DATE, fmt="date")}, ["schedule_id", "until"]),
])

APPS = [HOTEL, CLINIC, PET, LAUNDRY, SALON, EV, WIFI, EXPENSE, SCHOOL, APARTMENT, COURT, INSURANCE, DRIVE, GARDEN]


# ---- batch 2 -------------------------------------------------------------------------------------------------------------
A = "restaurant_booking"
RESTO = app(A, "Đặt bàn nhà hàng", "food", "Tìm nhà hàng, đặt bàn, đổi giờ hoặc huỷ đặt bàn", [
    tool(A, "search", "read", "Tìm nhà hàng theo tên, món hoặc khu vực; kết quả có các giờ còn bàn hôm nay và ngày mai (free_times)", {"query": s("Tên, món hoặc khu vực")}, returns="restaurants[]"),
    tool(A, "reserve", "write", "Đặt bàn vào một giờ còn bàn",
         {"restaurant_id": ref_id("nhà hàng", "restaurant_booking.search", "restaurant_booking.restaurants"), "time": s(ISO, fmt="date-time"),
          "party_size": i("Số người. " + INT, lo=1, hi=30), "note": s("Yêu cầu thêm (ghế trẻ em, phòng riêng...), chỉ khi người dùng nói", fuzzy=True)},
         ["restaurant_id", "time", "party_size"]),
    tool(A, "list_reservations", "read", "Các bàn đã đặt", {}, returns="reservations[]"),
    tool(A, "change_reservation", "write", "Đổi giờ hoặc số người của một bàn đã đặt",
         {"reservation_id": ref_id("đặt bàn", "restaurant_booking.list_reservations", "restaurant_booking.reservations"), "time": s(ISO, fmt="date-time"),
          "party_size": i(INT, lo=1, hi=30)}, ["reservation_id"]),
    tool(A, "cancel_reservation", "write", "Huỷ một bàn đã đặt", {"reservation_id": ref_id("đặt bàn", "restaurant_booking.list_reservations", "restaurant_booking.reservations")}, ["reservation_id"]),
])

A = "car_service"
CARSVC = app(A, "Bảo dưỡng xe", "transport", "Xe của người dùng, lịch bảo dưỡng định kỳ, đặt lịch garage", [
    tool(A, "list_vehicles", "read", "Các xe của người dùng với số km (odometer_km) và mốc bảo dưỡng kế tiếp (next_service_km)", {}, returns="vehicles[]"),
    tool(A, "list_garages", "read", "Garage gần đây và các gói dịch vụ (packages)", {"query": s("Tên hoặc khu vực")}, returns="garages[]"),
    tool(A, "book_service", "write", "Đặt lịch làm dịch vụ cho một xe tại garage",
         {"vehicle_id": ref_id("xe", "car_service.list_vehicles", "car_service.vehicles"), "garage_id": ref_id("garage", "car_service.list_garages", "car_service.garages"),
          "package_ids": arr({"type": "string"}, "ID các gói dịch vụ, lấy từ car_service.list_garages"), "time": s(ISO, fmt="date-time"),
          "pickup": b("Garage cử người tới lấy xe, chỉ khi người dùng yêu cầu")}, ["vehicle_id", "garage_id", "package_ids", "time"]),
    tool(A, "list_appointments", "read", "Các lịch đã đặt", {}, returns="appointments[]"),
    tool(A, "cancel_appointment", "write", "Huỷ lịch bảo dưỡng", {"appointment_id": ref_id("lịch", "car_service.list_appointments", "car_service.appointments")}, ["appointment_id"]),
])

A = "home_cleaning"
CLEAN = app(A, "Giúp việc theo giờ", "home", "Đặt người giúp việc theo giờ, dọn nhà, tổng vệ sinh", [
    tool(A, "get_packages", "read", "Các gói dọn dẹp và giá theo giờ", {}, returns="packages[]"),
    tool(A, "book", "write", "Đặt giúp việc",
         {"package_id": ref_id("gói", "home_cleaning.get_packages", "home_cleaning.packages"), "start": s(ISO, fmt="date-time"), "hours": i("Số giờ. " + INT, lo=2, hi=8),
          "repeat": s("Lặp lại; bỏ trống = một lần", enum=["none", "weekly", "biweekly"]), "note": s("Dặn thêm, chỉ khi người dùng nói", fuzzy=True)},
         ["package_id", "start", "hours"]),
    tool(A, "list_bookings", "read", "Các lịch giúp việc", {}, returns="bookings[]"),
    tool(A, "cancel", "write", "Huỷ một lịch giúp việc", {"booking_id": ref_id("lịch", "home_cleaning.list_bookings", "home_cleaning.bookings")}, ["booking_id"]),
    tool(A, "rate", "write", "Đánh giá buổi dọn đã xong",
         {"booking_id": ref_id("lịch", "home_cleaning.list_bookings", "home_cleaning.bookings"), "stars": i("Số sao. " + INT, lo=1, hi=5), "comment": s(fuzzy=True)}, ["booking_id", "stars"]),
])

A = "traffic_fines"
FINES = app(A, "Phạt nguội giao thông", "finance", "Tra cứu và nộp phạt vi phạm giao thông theo biển số", [
    tool(A, "list_vehicles", "read", "Các biển số đã đăng ký theo dõi", {}, returns="vehicles[]"),
    tool(A, "check_fines", "read", "Tra cứu vi phạm của một biển số", {"plate": s("Biển số, giữ đúng định dạng đã đăng ký, ví dụ 30H-123.45")}, ["plate"], returns="fines[]"),
    tool(A, "pay_fine", "write", "Nộp phạt một vi phạm", {"fine_id": ref_id("vi phạm", "traffic_fines.check_fines", "traffic_fines.fines"),
                                                          "method": s("Nguồn tiền; bỏ trống = thẻ đã liên kết", enum=["linked_card", "wallet"])}, ["fine_id"]),
    tool(A, "add_vehicle", "write", "Thêm một biển số để theo dõi", {"plate": s("Biển số"), "nickname": s("Tên gợi nhớ", fuzzy=True)}, ["plate"]),
])

A = "screen_time"
SCREEN = app(A, "Quản lý thời gian dùng máy của con", "family", "Giới hạn thời gian dùng thiết bị, khoá app, xem báo cáo", [
    tool(A, "get_report", "read", "Báo cáo thời gian dùng hôm nay của từng thiết bị/con", {}, returns="kids[]"),
    tool(A, "set_daily_limit", "write", "Đặt giới hạn số phút mỗi ngày cho một bé",
         {"kid_id": ref_id("bé", "screen_time.get_report", "screen_time.kids"), "minutes": i("Số phút. " + INT, lo=0, hi=720)}, ["kid_id", "minutes"]),
    tool(A, "block_app", "write", "Khoá hoặc mở một ứng dụng trên máy của bé",
         {"kid_id": ref_id("bé", "screen_time.get_report", "screen_time.kids"), "app_name": s("Tên ứng dụng, ví dụ YouTube"), "blocked": b("true = khoá, false = mở")},
         ["kid_id", "app_name", "blocked"]),
    tool(A, "grant_bonus", "write", "Cho thêm phút dùng máy hôm nay",
         {"kid_id": ref_id("bé", "screen_time.get_report", "screen_time.kids"), "minutes": i("Số phút thêm. " + INT, lo=5, hi=180)}, ["kid_id", "minutes"]),
])

A = "photo_album"
PHOTO = app(A, "Album ảnh", "personal", "Album ảnh: tìm album, tạo album, chia sẻ, thêm ảnh theo ngày", [
    tool(A, "list_albums", "read", "Các album ảnh", {"query": s("Tên album")}, returns="albums[]"),
    tool(A, "create_album", "write", "Tạo album mới, có thể gom ảnh chụp trong một khoảng ngày",
         {"title": s(fuzzy=True), "date_from": s(DATE, fmt="date"), "date_to": s(DATE, fmt="date")}, ["title"]),
    tool(A, "share_album", "write", "Chia sẻ album cho một liên hệ",
         {"album_id": ref_id("album", "photo_album.list_albums", "photo_album.albums"), "contact_id": s("ID liên hệ, lấy từ contacts.search", ref="contacts.contacts")},
         ["album_id", "contact_id"]),
    tool(A, "delete_album", "write", "Xoá album (ảnh vẫn giữ trong thư viện)", {"album_id": ref_id("album", "photo_album.list_albums", "photo_album.albums")}, ["album_id"]),
])

A = "charity"
CHARITY = app(A, "Quyên góp từ thiện", "finance", "Các chiến dịch quyên góp; ủng hộ một lần hoặc hằng tháng", [
    tool(A, "list_campaigns", "read", "Các chiến dịch đang mở", {"query": s("Tên hoặc chủ đề")}, returns="campaigns[]"),
    tool(A, "donate", "write", "Ủng hộ một chiến dịch",
         {"campaign_id": ref_id("chiến dịch", "charity.list_campaigns", "charity.campaigns"), "amount": i("Số tiền (đồng). " + INT, lo=10000),
          "monthly": b("Ủng hộ hằng tháng, chỉ khi người dùng nói"), "anonymous": b("Ẩn danh, chỉ khi người dùng nói")}, ["campaign_id", "amount"]),
    tool(A, "list_donations", "read", "Các lần đã ủng hộ", {}, returns="donations[]"),
    tool(A, "stop_monthly", "write", "Dừng ủng hộ hằng tháng", {"donation_id": ref_id("khoản ủng hộ", "charity.list_donations", "charity.donations")}, ["donation_id"]),
])

A = "streaming"
STREAM = app(A, "Xem phim trực tuyến", "entertainment", "Tìm phim, danh sách xem sau, tải về xem offline, hồ sơ trẻ em", [
    tool(A, "search_titles", "read", "Tìm phim/chương trình theo tên hoặc thể loại", {"query": s("Tên hoặc thể loại")}, ["query"], returns="titles[]"),
    tool(A, "get_watchlist", "read", "Danh sách xem sau của một hồ sơ", {"profile": s(enum=["main", "kids"])}, returns="watchlist[]"),
    tool(A, "add_to_watchlist", "write", "Thêm vào danh sách xem sau",
         {"title_id": ref_id("phim", "streaming.search_titles", "streaming.titles"), "profile": s("Hồ sơ; bỏ trống = main", enum=["main", "kids"])}, ["title_id"]),
    tool(A, "remove_from_watchlist", "write", "Bỏ khỏi danh sách xem sau",
         {"title_id": ref_id("phim", "streaming.get_watchlist", "streaming.titles"), "profile": s(enum=["main", "kids"])}, ["title_id"]),
    tool(A, "download", "write", "Tải phim/tập phim về máy",
         {"title_id": ref_id("phim", "streaming.search_titles", "streaming.titles"), "episode": i("Tập số, chỉ với phim bộ. " + INT, lo=1),
          "quality": s("Chất lượng; bỏ trống = standard", enum=["standard", "high"])}, ["title_id"]),
])

A = "health_monitor"
HEALTH = app(A, "Sổ sức khoẻ gia đình", "health", "Ghi chỉ số huyết áp, đường huyết, cân nặng của các thành viên; xem lịch sử", [
    tool(A, "list_members", "read", "Thành viên gia đình được theo dõi", {}, returns="members[]"),
    tool(A, "log_reading", "write", "Ghi một chỉ số đo",
         {"member_id": ref_id("thành viên", "health_monitor.list_members", "health_monitor.members"),
          "type": s("blood_pressure = huyết áp (value = tâm thu, value2 = tâm trương), glucose = đường huyết (mmol/L), weight = cân nặng (kg), heart_rate = nhịp tim",
                    enum=["blood_pressure", "glucose", "weight", "heart_rate"]),
          "value": n("Giá trị đo"), "value2": n("Chỉ số thứ hai (tâm trương với huyết áp)"), "note": s(fuzzy=True)}, ["member_id", "type", "value"]),
    tool(A, "get_history", "read", "Lịch sử chỉ số của một thành viên",
         {"member_id": ref_id("thành viên", "health_monitor.list_members", "health_monitor.members"), "type": s(enum=["blood_pressure", "glucose", "weight", "heart_rate"])},
         ["member_id"], returns="readings[]"),
    tool(A, "share_report", "write", "Gửi báo cáo chỉ số gần đây cho bác sĩ gia đình qua email",
         {"member_id": ref_id("thành viên", "health_monitor.list_members", "health_monitor.members"), "email": s("Email bác sĩ"), "days": i("Số ngày gần nhất. " + INT, lo=1, hi=90)},
         ["member_id", "email"]),
])

A = "public_services"
GOV = app(A, "Dịch vụ công", "government", "Hồ sơ dịch vụ công: tra cứu trạng thái, đặt lịch hẹn tại cơ quan, nộp lệ phí", [
    tool(A, "list_applications", "read", "Các hồ sơ đã nộp và trạng thái", {}, returns="applications[]"),
    tool(A, "list_offices", "read", "Cơ quan tiếp nhận và các khung giờ hẹn còn trống", {"service": s("passport = hộ chiếu, id_card = căn cước, residence = cư trú, land = đất đai",
                                                                                                    enum=["passport", "id_card", "residence", "land"])}, ["service"], returns="offices[]"),
    tool(A, "book_slot", "write", "Đặt lịch hẹn tại cơ quan cho một dịch vụ",
         {"office_id": ref_id("cơ quan", "public_services.list_offices", "public_services.offices"), "service": s(enum=["passport", "id_card", "residence", "land"]),
          "time": s(ISO, fmt="date-time")}, ["office_id", "service", "time"]),
    tool(A, "pay_fee", "write", "Nộp lệ phí của một hồ sơ", {"application_id": ref_id("hồ sơ", "public_services.list_applications", "public_services.applications")}, ["application_id"]),
])

A = "car_remote"
CAR = app(A, "Điều khiển ô tô từ xa", "transport", "Xem trạng thái xe, khoá/mở cửa, bật điều hoà trước, tìm vị trí xe", [
    tool(A, "get_status", "read", "Trạng thái xe: khoá cửa, nhiệt độ trong xe, nhiên liệu/pin, vị trí", {}, returns="car"),
    tool(A, "lock", "write", "Khoá hoặc mở khoá cửa xe", {"locked": b("true = khoá, false = mở")}, ["locked"]),
    tool(A, "climate_start", "write", "Bật điều hoà trước khi lên xe",
         {"temperature": i("Nhiệt độ (độ C). " + INT, lo=16, hi=30), "minutes": i("Chạy trong bao nhiêu phút; bỏ trống = 10. " + INT, lo=5, hi=30)}, ["temperature"]),
    tool(A, "climate_stop", "write", "Tắt điều hoà đang chạy từ xa", {}),
    tool(A, "honk_flash", "write", "Bấm còi và nháy đèn để tìm xe", {"mode": s("both = còi và đèn, flash = chỉ nháy đèn", enum=["both", "flash"])}, ["mode"]),
])

A = "baby_tracker"
BABY = app(A, "Nhật ký em bé", "family", "Ghi cữ bú, giấc ngủ, thay tã của bé; xem lần gần nhất", [
    tool(A, "list_babies", "read", "Các bé", {}, returns="babies[]"),
    tool(A, "last_events", "read", "Các sự kiện gần nhất của bé", {"baby_id": ref_id("bé", "baby_tracker.list_babies", "baby_tracker.babies"),
                                                                   "type": s(enum=["feed", "sleep", "diaper"])}, ["baby_id"], returns="events[]"),
    tool(A, "log_feed", "write", "Ghi một cữ bú",
         {"baby_id": ref_id("bé", "baby_tracker.list_babies", "baby_tracker.babies"), "method": s("bottle = bình, breast = bú mẹ", enum=["bottle", "breast"]),
          "amount_ml": i("Lượng sữa (ml), với bú bình. " + INT, lo=0, hi=400), "time": s(ISO + "; bỏ trống = bây giờ", fmt="date-time")}, ["baby_id", "method"]),
    tool(A, "log_sleep", "write", "Ghi một giấc ngủ",
         {"baby_id": ref_id("bé", "baby_tracker.list_babies", "baby_tracker.babies"), "start": s(ISO, fmt="date-time"), "end": s(ISO, fmt="date-time")}, ["baby_id", "start", "end"]),
    tool(A, "log_diaper", "write", "Ghi một lần thay tã",
         {"baby_id": ref_id("bé", "baby_tracker.list_babies", "baby_tracker.babies"), "kind": s(enum=["wet", "dirty", "both"])}, ["baby_id", "kind"]),
])

A = "savings_goals"
GOALS = app(A, "Mục tiêu tiết kiệm", "finance", "Heo đất theo mục tiêu: tạo mục tiêu, góp tiền, rút tiền", [
    tool(A, "list_goals", "read", "Các mục tiêu với số đã góp (saved) và đích (target)", {}, returns="goals[]"),
    tool(A, "create_goal", "write", "Tạo mục tiêu tiết kiệm mới",
         {"name": s(fuzzy=True), "target": i("Số tiền mục tiêu (đồng). " + INT, lo=100000), "deadline": s(DATE, fmt="date")}, ["name", "target"]),
    tool(A, "deposit", "write", "Góp tiền vào một mục tiêu", {"goal_id": ref_id("mục tiêu", "savings_goals.list_goals", "savings_goals.goals"), "amount": i("Số tiền (đồng). " + INT, lo=1000)},
         ["goal_id", "amount"]),
    tool(A, "withdraw", "write", "Rút tiền khỏi một mục tiêu", {"goal_id": ref_id("mục tiêu", "savings_goals.list_goals", "savings_goals.goals"), "amount": i("Số tiền (đồng). " + INT, lo=1000)},
         ["goal_id", "amount"]),
    tool(A, "set_auto_deposit", "write", "Tự động góp hằng tháng vào ngày cố định",
         {"goal_id": ref_id("mục tiêu", "savings_goals.list_goals", "savings_goals.goals"), "amount": i("Số tiền mỗi tháng (đồng). " + INT, lo=1000),
          "day_of_month": i("Ngày trong tháng. " + INT, lo=1, hi=28)}, ["goal_id", "amount", "day_of_month"]),
])

APPS = APPS + [RESTO, CARSVC, CLEAN, FINES, SCREEN, PHOTO, CHARITY, STREAM, HEALTH, GOV, CAR, BABY, GOALS]
