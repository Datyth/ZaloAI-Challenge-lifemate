"""Tasks: pet_care, laundry, salon, wifi_router, smart_garden. Time anchor Thursday 2026-11-12 unless stated."""
from _common import ASK, C, CONTACTS, REMINDERS, SMS, T

NOW = "2026-11-12T09:00:00+07:00"
EVE = "2026-11-12T19:30:00+07:00"

PET = {"pets": [
    {"id": "pet_01", "name": "Mực", "species": "dog", "breed": "Poodle", "weight_kg": 4.2},
    {"id": "pet_02", "name": "Bông", "species": "cat", "breed": "Mèo Anh lông ngắn", "weight_kg": 3.8},
], "logs": [
    {"id": "log_01", "pet_id": "pet_01", "type": "vaccine", "time": "2026-05-10T10:00:00+07:00", "amount": None, "note": "dại"},
    {"id": "log_02", "pet_id": "pet_02", "type": "feeding", "time": "2026-11-12T07:00:00+07:00", "amount": 40, "note": None},
], "schedules": []}
P = {"pet_care": PET}

LAUNDRY = {"default_address": "Căn hộ đăng ký trong tài khoản", "services": [
    {"id": "sv_01", "name": "Giặt sấy quần áo", "unit": "kg", "price": 18000},
    {"id": "sv_02", "name": "Giặt hấp áo vest", "unit": "chiếc", "price": 60000},
    {"id": "sv_03", "name": "Giặt chăn mền", "unit": "chiếc", "price": 80000},
    {"id": "sv_04", "name": "Giặt giày", "unit": "đôi", "price": 50000},
], "orders": [
    {"id": "ord_01", "items": [{"service_id": "sv_01", "quantity": 5}], "pickup_time": "2026-11-13T08:00:00+07:00", "address": "Căn hộ đăng ký trong tài khoản",
     "express": False, "total": 90000, "status": "scheduled"},
]}
L = {"laundry": LAUNDRY}

SALON = {"salons": [
    {"id": "sl_01", "name": "Tóc Đẹp Hair Salon", "address": "45 Nguyễn Trãi, Thanh Xuân, Hà Nội",
     "services": [{"id": "s_cut", "name": "Cắt tóc nam", "duration_min": 30, "price": 120000}, {"id": "s_wash", "name": "Gội đầu dưỡng sinh", "duration_min": 45, "price": 150000},
                  {"id": "s_dye", "name": "Nhuộm tóc", "duration_min": 90, "price": 450000}],
     "stylists": [{"id": "st_01", "name": "Anh Tùng"}, {"id": "st_02", "name": "Chị Mai"}]},
    {"id": "sl_02", "name": "Lotus Nail & Spa", "address": "12 Lý Thường Kiệt, Hoàn Kiếm, Hà Nội",
     "services": [{"id": "n_mani", "name": "Làm móng tay", "duration_min": 60, "price": 200000}, {"id": "n_pedi", "name": "Làm móng chân", "duration_min": 60, "price": 220000},
                  {"id": "n_massage", "name": "Massage chân", "duration_min": 45, "price": 250000}],
     "stylists": [{"id": "st_03", "name": "Chị Thảo"}]},
], "bookings": [
    {"id": "sb_01", "salon_id": "sl_01", "service_ids": ["s_cut"], "time": "2026-11-14T10:00:00+07:00", "stylist_id": "st_01", "total": 120000, "status": "booked"},
]}
S = {"salon": SALON}
SC = {"salon": SALON, "contacts": CONTACTS, "sms": SMS}

WIFI = {"network": {"ssid": "NhaMinh_5G", "password": "matkhaucu123", "guest_enabled": False, "guest_password": None}, "devices": [
    {"id": "dev_01", "name": "iPad của Bi", "owner": "Bi", "blocked": False},
    {"id": "dev_02", "name": "Laptop bố", "owner": "bố", "blocked": False},
    {"id": "dev_03", "name": "Smart TV phòng khách", "owner": "chung", "blocked": False},
    {"id": "dev_04", "name": "Điện thoại lạ Galaxy-A12", "owner": None, "blocked": False},
], "schedules": []}
W = {"wifi_router": WIFI}

GARDEN = {"zones": [
    {"id": "z_01", "name": "Ban công", "plant": "hoa giấy", "moisture": 22, "last_watered": "2026-11-11T06:00:00+07:00"},
    {"id": "z_02", "name": "Sân trước", "plant": "cỏ", "moisture": 48, "last_watered": "2026-11-12T06:00:00+07:00"},
    {"id": "z_03", "name": "Vườn rau", "plant": "rau cải", "moisture": 35, "last_watered": "2026-11-11T17:00:00+07:00"},
], "schedules": [{"id": "gs_01", "zone_id": "z_02", "time": "06:00", "duration_min": 10, "days": ["all"], "paused_until": None}]}
G = {"smart_garden": GARDEN}

TASKS = [
    # pet_care
    T("pet_log_feed", EVE, "Ghi lại là mới cho Mực ăn tám mươi gam hạt nha.", [C("pet_care.list_pets"), C("pet_care.log_activity", pet_id="pet_01", type="feeding", amount=80)], P),
    T("pet_log_walk", EVE, "Vừa dắt Mực đi dạo ba mươi phút xong, ghi vào giúp chị.", [C("pet_care.list_pets"), C("pet_care.log_activity", pet_id="pet_01", type="walk", amount=30)], P),
    T("pet_schedule_vaccine", NOW, "Đặt lịch tiêm phòng cho con Bông chín giờ sáng thứ bảy này.",
      [C("pet_care.list_pets"), C("pet_care.schedule_care", pet_id="pet_02", type="vaccine", time="2026-11-14T09:00:00+07:00")], P),
    T("pet_schedule_daily", NOW, "Mỗi ngày bảy giờ tối nhắc tắm cho Mực, à không, cho Mực uống thuốc, bắt đầu từ tối nay.",
      [C("pet_care.list_pets"), C("pet_care.schedule_care", pet_id="pet_01", type="medicine", time="2026-11-12T19:00:00+07:00", repeat="daily")], P),
    T("pet_weight", NOW, "Bông nặng bốn ký mốt rồi, cập nhật lại cân nặng.", [C("pet_care.list_pets"), C("pet_care.update_weight", pet_id="pet_02", weight_kg=4.1)], P),
    T("pet_fed_check", NOW, "Sáng nay ai cho Bông ăn chưa nhỉ?", [C("pet_care.list_pets"), C("pet_care.get_logs", pet_id="pet_02", type="feeding")], P),
    T("pet_missing_which", NOW, "Đặt lịch đi khám thú y chiều mai ba giờ.", [C("pet_care.list_pets"), ASK("pet_care.schedule_care", "pet_id")], P),
    T("pet_missing_time", NOW, "Hẹn lịch tắm cho Mực giúp em.", [C("pet_care.list_pets"), ASK("pet_care.schedule_care", "time")], P),
    # laundry
    T("laundry_kg", NOW, "Đặt giặt sấy sáu ký đồ, lấy lúc năm giờ chiều nay nhé.",
      [C("laundry.list_services"), C("laundry.create_order", items=[{"service_id": "sv_01", "quantity": 6}], pickup_time="2026-11-12T17:00:00+07:00")], L),
    T("laundry_multi", NOW, "Mai tám giờ sáng qua lấy hai cái áo vest với một cái chăn đi giặt giùm anh.",
      [C("laundry.list_services"), C("laundry.create_order", items=[{"service_id": "sv_02", "quantity": 2}, {"service_id": "sv_03", "quantity": 1}],
                                     pickup_time="2026-11-13T08:00:00+07:00")], L),
    T("laundry_express_addr", NOW, "Giặt nhanh trong ngày ba đôi giày, lấy ở 18 Ngõ Huế lúc mười giờ sáng nay.",
      [C("laundry.list_services"), C("laundry.create_order", items=[{"service_id": "sv_04", "quantity": 3}], pickup_time="2026-11-12T10:00:00+07:00",
                                     address="18 Ngõ Huế", express=True)], L),
    T("laundry_missing_qty", NOW, "Đặt giặt chăn cho chị chiều mai bốn giờ.", [C("laundry.list_services"), ASK("laundry.create_order", "items")], L),
    T("laundry_cancel", NOW, "Đơn giặt sáng mai hủy đi em, tuần sau chị gửi.", [C("laundry.list_orders"), C("laundry.cancel_order", order_id="ord_01")], L),
    T("laundry_price", NOW, "Giặt hấp vest bao nhiêu tiền một cái?", [C("laundry.list_services")], L),
    # salon
    T("salon_cut", NOW, "Đặt cắt tóc nam ở Tóc Đẹp lúc sáu giờ chiều mai.",
      [C("salon.search_salons", query="Tóc Đẹp"), C("salon.get_salon", salon_id="sl_01"),
       C("salon.book", salon_id="sl_01", service_ids=["s_cut"], time="2026-11-13T18:00:00+07:00")], S),
    T("salon_combo_stylist", NOW, "Chủ nhật này hai giờ chiều em muốn nhuộm tóc với gội đầu ở Tóc Đẹp, chọn chị Mai làm nha.",
      [C("salon.search_salons", query="Tóc Đẹp"), C("salon.get_salon", salon_id="sl_01"),
       C("salon.book", salon_id="sl_01", service_ids=["s_dye", "s_wash"], time="2026-11-15T14:00:00+07:00", stylist_id="st_02")], S),
    T("salon_nail", NOW, "Book làm móng tay với móng chân ở Lotus tối nay bảy giờ.",
      [C("salon.search_salons", query="Lotus"), C("salon.get_salon", salon_id="sl_02"),
       C("salon.book", salon_id="sl_02", service_ids=["n_mani", "n_pedi"], time="2026-11-12T19:00:00+07:00")], S),
    T("salon_cancel", NOW, "Thứ bảy anh bận rồi, hủy lịch cắt tóc đi.", [C("salon.list_bookings"), C("salon.cancel_booking", booking_id="sb_01")], S),
    T("salon_missing_time", NOW, "Đặt massage chân ở Lotus cho chị.", [C("salon.search_salons", query="Lotus"), C("salon.get_salon", salon_id="sl_02"), ASK("salon.book", "time")], S),
    T("salon_book_sms", NOW, "Đặt gội đầu dưỡng sinh ở Tóc Đẹp năm giờ chiều nay rồi nhắn Linh là tối nay đi ăn trễ chút nha.",
      [C("salon.search_salons", query="Tóc Đẹp"), C("salon.get_salon", salon_id="sl_01"),
       C("salon.book", salon_id="sl_01", service_ids=["s_wash"], time="2026-11-12T17:00:00+07:00"),
       C("contacts.search", query="Linh"), C("sms.send", contact_id="c_06", content="Tối nay đi ăn trễ chút nha")], SC),
    # wifi_router
    T("wifi_block_unknown", NOW, "Có cái máy lạ nào đang xài wifi nhà mình thì chặn nó luôn.",
      [C("wifi_router.get_status"), C("wifi_router.block_device", device_id="dev_04", blocked=True)], W),
    T("wifi_kid_schedule", NOW, "Cài cho iPad của Bi mất mạng từ chín giờ tối tới sáu giờ sáng các ngày trong tuần, thứ hai tới thứ sáu.",
      [C("wifi_router.get_status"), C("wifi_router.set_schedule", device_id="dev_01", off_from="21:00", off_to="06:00", days=["mon", "tue", "wed", "thu", "fri"])], W),
    T("wifi_guest_on", NOW, "Bật mạng khách cho khách đến chơi, mật khẩu là chaomung2026.", [C("wifi_router.set_guest_network", enabled=True, password="chaomung2026")], W),
    T("wifi_guest_off", NOW, "Khách về rồi, tắt mạng khách đi.", [C("wifi_router.set_guest_network", enabled=False)], W),
    T("wifi_unblock", NOW, "Mở lại mạng cho cái laptop của bố.", [C("wifi_router.get_status"), C("wifi_router.block_device", device_id="dev_02", blocked=False)], W),
    T("wifi_change_pw_missing", NOW, "Đổi mật khẩu wifi giùm anh.", [ASK("wifi_router.change_password", "new_password")], W),
    T("wifi_change_pw", NOW, "Đổi pass wifi thành nhaminh@2026 nhé.", [C("wifi_router.change_password", new_password="nhaminh@2026")], W),
    T("wifi_who", NOW, "Đang có những máy nào kết nối wifi vậy?", [C("wifi_router.get_status")], W),
    # smart_garden
    T("garden_dry_zone", NOW, "Khu nào khô dưới ba mươi phần trăm thì tưới ngay mười phút.",
      [C("smart_garden.get_status"), C("smart_garden.water_now", zone_id="z_01", duration_min=10)], G),
    T("garden_veg", NOW, "Tưới vườn rau năm phút đi em.", [C("smart_garden.get_status"), C("smart_garden.water_now", zone_id="z_03", duration_min=5)], G),
    T("garden_schedule", NOW, "Đặt lịch tưới ban công năm giờ rưỡi chiều mỗi ngày, mỗi lần tám phút.",
      [C("smart_garden.get_status"), C("smart_garden.set_schedule", zone_id="z_01", time="17:30", duration_min=8)], G),
    T("garden_pause_rain", NOW, "Mai với mốt trời mưa, tạm dừng lịch tưới sân trước tới hết thứ bảy nhé.",
      [C("smart_garden.get_status"), C("smart_garden.pause_schedule", schedule_id="gs_01", until="2026-11-14")], G),
    T("garden_missing_minutes", NOW, "Tưới sân trước ngay giúp anh.", [C("smart_garden.get_status"), ASK("smart_garden.water_now", "duration_min")], G),
    T("garden_weekend", NOW, "Vườn rau tưới lúc sáu giờ sáng thứ bảy chủ nhật, mười lăm phút.",
      [C("smart_garden.get_status"), C("smart_garden.set_schedule", zone_id="z_03", time="06:00", duration_min=15, days=["sat", "sun"])], G),
    T("garden_out_of_scope", NOW, "Bón phân cho vườn rau giúp tôi.", [], G),
]
