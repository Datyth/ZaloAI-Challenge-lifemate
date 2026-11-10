"""Tasks for the batch-2 synthetic apps. Anchors: Mon 2026-11-23 20:00 (EVE), Thu 2026-11-19 08:30 (MORN), Sat 2026-11-28 10:00 (SAT)."""
from _common import ASK, C, CALENDAR, CONTACTS, SMS, T

EVE = "2026-11-23T20:00:00+07:00"
MORN = "2026-11-19T08:30:00+07:00"
SAT = "2026-11-28T10:00:00+07:00"

RESTO = {"restaurants": [
    {"id": "rs_01", "name": "Lẩu Nấm Ashima", "cuisine": "lẩu nấm", "address": "44 Hàng Bông, Hoàn Kiếm", "max_party": 20,
     "free_times": ["2026-11-23T19:00", "2026-11-24T18:30", "2026-11-24T19:00", "2026-11-24T19:30"]},
    {"id": "rs_02", "name": "Pizza 4P's Bảo Khánh", "cuisine": "pizza Ý", "address": "11B Bảo Khánh, Hoàn Kiếm", "max_party": 8,
     "free_times": ["2026-11-24T12:00", "2026-11-24T18:00", "2026-11-24T20:00"]},
    {"id": "rs_03", "name": "Quán Bún Chả Hương Liên", "cuisine": "bún chả", "address": "24 Lê Văn Hưu, Hai Bà Trưng", "max_party": 6,
     "free_times": ["2026-11-24T11:30", "2026-11-24T12:00"]},
], "reservations": [{"id": "rsv_01", "restaurant_id": "rs_02", "time": "2026-11-25T19:00:00+07:00", "party_size": 4, "note": None, "status": "booked"}]}
R = {"restaurant_booking": RESTO}
RC = {"restaurant_booking": RESTO, "contacts": CONTACTS, "sms": SMS, "calendar": CALENDAR}

CARS = {"vehicles": [{"id": "car_01", "name": "Mazda CX-5", "plate": "30H-123.45", "odometer_km": 39650, "next_service_km": 40000},
                     {"id": "car_02", "name": "Honda Vision", "plate": "29B1-678.90", "odometer_km": 12100, "next_service_km": 15000}],
        "garages": [{"id": "gr_01", "name": "Mazda Long Biên", "address": "Nguyễn Văn Linh, Long Biên",
                     "packages": [{"id": "pk_40k", "name": "Bảo dưỡng cấp 40.000 km", "price": 4200000}, {"id": "pk_wash", "name": "Rửa xe, hút bụi", "price": 150000},
                                  {"id": "pk_tire", "name": "Đảo lốp, cân bằng động", "price": 400000}]},
                    {"id": "gr_02", "name": "Honda Head Minh Long", "address": "Kim Mã, Ba Đình",
                     "packages": [{"id": "pk_oil", "name": "Thay dầu xe máy", "price": 120000}, {"id": "pk_check", "name": "Kiểm tra tổng quát xe máy", "price": 80000}]}],
        "appointments": []}
CS = {"car_service": CARS}

CLEAN = {"packages": [{"id": "pkg_basic", "name": "Dọn nhà theo giờ", "price_per_hour": 90000}, {"id": "pkg_deep", "name": "Tổng vệ sinh", "price_per_hour": 150000},
                      {"id": "pkg_sofa", "name": "Giặt sofa, nệm", "price_per_hour": 200000}],
         "bookings": [{"id": "hc_01", "package_id": "pkg_basic", "start": "2026-11-21T08:00:00+07:00", "hours": 3, "repeat": "weekly", "total": 270000, "status": "done", "rating": None},
                      {"id": "hc_02", "package_id": "pkg_basic", "start": "2026-11-28T08:00:00+07:00", "hours": 3, "repeat": "weekly", "total": 270000, "status": "booked", "rating": None}]}
HC = {"home_cleaning": CLEAN}

FINES = {"vehicles": [{"id": "veh_01", "plate": "30H-123.45", "nickname": "xe ô tô"}, {"id": "veh_02", "plate": "29B1-678.90", "nickname": "xe máy của vợ"}],
         "fines": [{"id": "fn_01", "plate": "30H-123.45", "violation": "Vượt đèn đỏ", "place": "Ngã tư Kim Mã - Liễu Giai", "date": "2026-11-02", "amount": 4000000, "paid": False},
                   {"id": "fn_02", "plate": "30H-123.45", "violation": "Quá tốc độ 8 km/h", "place": "Đại lộ Thăng Long", "date": "2026-10-15", "amount": 1000000, "paid": True}]}
F = {"traffic_fines": FINES}

SCREEN = {"kids": [{"id": "kid_01", "name": "Bi", "device": "iPad", "used_today_min": 95, "daily_limit_min": 120, "blocked_apps": ["TikTok"], "bonus_today_min": 0},
                   {"id": "kid_02", "name": "Bông", "device": "Điện thoại Samsung", "used_today_min": 140, "daily_limit_min": 90, "blocked_apps": [], "bonus_today_min": 0}]}
SC = {"screen_time": SCREEN}

PHOTO = {"albums": [{"id": "alb_01", "title": "Sinh nhật bé Na", "photo_ids": ["p_01", "p_02"], "shared_with": []},
                    {"id": "alb_02", "title": "Du lịch Sa Pa 2025", "photo_ids": ["p_03"], "shared_with": ["c_03"]}],
         "photos": [{"id": "p_01", "date": "2026-10-04"}, {"id": "p_02", "date": "2026-10-04"}, {"id": "p_03", "date": "2025-12-20"},
                    {"id": "p_04", "date": "2026-11-21"}, {"id": "p_05", "date": "2026-11-22"}, {"id": "p_06", "date": "2026-11-22"}]}
PH = {"photo_album": PHOTO, "contacts": CONTACTS}

CHARITY = {"campaigns": [{"id": "cp_01", "title": "Áo ấm cho em vùng cao", "topic": "giáo dục, trẻ em", "organizer": "Quỹ Sưởi Ấm", "raised": 152000000},
                         {"id": "cp_02", "title": "Hỗ trợ bà con vùng lũ miền Trung", "topic": "thiên tai", "organizer": "Hội Chữ thập đỏ", "raised": 980000000},
                         {"id": "cp_03", "title": "Bữa cơm 0 đồng bệnh viện K", "topic": "y tế", "organizer": "Nhóm Thiện Tâm", "raised": 45000000}],
           "donations": [{"id": "don_01", "campaign_id": "cp_03", "amount": 200000, "monthly": True, "anonymous": False, "active": True}]}
CH = {"charity": CHARITY}

STREAM = {"titles": [{"id": "tt_01", "name": "Mắt Biếc", "type": "movie", "genre": "tình cảm"},
                     {"id": "tt_02", "name": "Gia Đình Là Số Một", "type": "series", "genre": "hài, gia đình", "episodes": 40},
                     {"id": "tt_03", "name": "Doraemon: Nobita và vùng đất lý tưởng", "type": "movie", "genre": "hoạt hình"},
                     {"id": "tt_04", "name": "Hành Tinh Xanh", "type": "series", "genre": "tài liệu", "episodes": 7}],
          "watchlist": {"main": ["tt_01"], "kids": []}, "downloads": []}
ST = {"streaming": STREAM}

HEALTH = {"members": [{"id": "mb_01", "name": "Bố", "age": 68}, {"id": "mb_02", "name": "Mẹ", "age": 64}, {"id": "mb_03", "name": "Tôi", "age": 36}],
          "readings": [{"id": "rd_01", "member_id": "mb_01", "type": "blood_pressure", "value": 145, "value2": 92, "note": None, "time": "2026-11-18T07:00:00+07:00"},
                       {"id": "rd_02", "member_id": "mb_02", "type": "glucose", "value": 7.8, "value2": None, "note": "sau ăn", "time": "2026-11-18T12:30:00+07:00"}],
          "shares": []}
HM = {"health_monitor": HEALTH}

GOV = {"applications": [{"id": "app_01", "service": "passport", "status": "processing", "fee": 200000, "fee_paid": False},
                        {"id": "app_02", "service": "residence", "status": "done", "fee": 0, "fee_paid": False}],
       "offices": [{"id": "off_01", "name": "Phòng Xuất nhập cảnh Công an Hà Nội", "services": ["passport"], "free_slots": ["2026-11-20T08:00", "2026-11-24T14:00"]},
                   {"id": "off_02", "name": "Công an phường Dịch Vọng", "services": ["id_card", "residence"], "free_slots": ["2026-11-19T14:00", "2026-11-21T08:30"]}],
       "appointments": []}
GV = {"public_services": GOV}
GVC = {"public_services": GOV, "calendar": CALENDAR}

CAR = {"car": {"model": "VinFast VF 6", "locked": False, "cabin_temp_c": 34, "battery_percent": 64, "location": "Hầm B2 toà Keangnam", "climate": {"on": False}}}
CR = {"car_remote": CAR}

BABY = {"babies": [{"id": "bb_01", "name": "Su", "birth_date": "2026-07-02"}],
        "events": [{"id": "ev_01", "baby_id": "bb_01", "type": "feed", "method": "bottle", "amount_ml": 120, "time": "2026-11-23T17:30:00+07:00"},
                   {"id": "ev_02", "baby_id": "bb_01", "type": "diaper", "kind": "wet", "time": "2026-11-23T18:10:00+07:00"}]}
BB = {"baby_tracker": BABY}

GOALS = {"goals": [{"id": "goal_01", "name": "Mua xe máy cho con", "target": 40000000, "deadline": "2027-06-01", "saved": 12500000, "auto": None},
                   {"id": "goal_02", "name": "Du lịch Phú Quốc", "target": 15000000, "deadline": "2027-04-30", "saved": 9000000, "auto": None}]}
SG = {"savings_goals": GOALS}

TASKS = [
    # restaurant_booking
    T("resto_book", EVE, "Đặt bàn lẩu nấm Ashima bảy giờ tối mai cho sáu người nhé.",
      [C("restaurant_booking.search", query="Ashima"), C("restaurant_booking.reserve", restaurant_id="rs_01", time="2026-11-24T19:00:00+07:00", party_size=6)], R),
    T("resto_book_note", EVE, "Trưa mai mười hai giờ đặt quán bún chả Hương Liên cho ba người, có một ghế trẻ em.",
      [C("restaurant_booking.search", query="Hương Liên"),
       C("restaurant_booking.reserve", restaurant_id="rs_03", time="2026-11-24T12:00:00+07:00", party_size=3, note="một ghế trẻ em")], R),
    T("resto_change", EVE, "Bàn pizza thứ tư thêm hai người nữa nha, thành sáu.",
      [C("restaurant_booking.list_reservations"), C("restaurant_booking.change_reservation", reservation_id="rsv_01", party_size=6)], R),
    T("resto_cancel", EVE, "Hủy bàn ở 4P's tối thứ tư đi em.", [C("restaurant_booking.list_reservations"), C("restaurant_booking.cancel_reservation", reservation_id="rsv_01")], R),
    T("resto_missing_size", EVE, "Đặt bàn Pizza 4P's tám giờ tối mai.", [C("restaurant_booking.search", query="4P's"), ASK("restaurant_booking.reserve", "party_size")], R),
    T("resto_too_big", EVE, "Đặt bàn bún chả Hương Liên trưa mai mười giờ rưỡi... à mười một rưỡi, cho mười người.", [C("restaurant_booking.search", query="Hương Liên")], R),
    T("resto_book_invite", EVE, "Đặt Ashima bảy rưỡi tối mai cho bốn người, rồi thêm vào lịch và nhắn Linh tối mai đi ăn lẩu nấm nhé.",
      [C("restaurant_booking.search", query="Ashima"), C("restaurant_booking.reserve", restaurant_id="rs_01", time="2026-11-24T19:30:00+07:00", party_size=4),
       C("calendar.create_event", title="Ăn lẩu nấm Ashima", start="2026-11-24T19:30:00+07:00", location="44 Hàng Bông, Hoàn Kiếm"),
       C("contacts.search", query="Linh"), C("sms.send", contact_id="c_06", content="Tối mai đi ăn lẩu nấm Ashima lúc 7 rưỡi nhé")], RC),
    # car_service
    T("carsvc_due", MORN, "Xe ô tô sắp tới kỳ bảo dưỡng chưa, nếu còn dưới năm trăm cây số thì đặt bảo dưỡng ở Mazda Long Biên tám giờ sáng thứ bảy.",
      [C("car_service.list_vehicles"), C("car_service.list_garages", query="Mazda Long Biên"),
       C("car_service.book_service", vehicle_id="car_01", garage_id="gr_01", package_ids=["pk_40k"], time="2026-11-21T08:00:00+07:00")], CS),
    T("carsvc_moto", MORN, "Mang xe Vision đi thay dầu ở Honda Minh Long chiều nay ba giờ.",
      [C("car_service.list_vehicles"), C("car_service.list_garages", query="Minh Long"),
       C("car_service.book_service", vehicle_id="car_02", garage_id="gr_02", package_ids=["pk_oil"], time="2026-11-19T15:00:00+07:00")], CS),
    T("carsvc_pickup", MORN, "Đặt rửa xe với đảo lốp cho chiếc Mazda ở Long Biên mười giờ sáng mai, cho người tới lấy xe tận nơi.",
      [C("car_service.list_vehicles"), C("car_service.list_garages", query="Long Biên"),
       C("car_service.book_service", vehicle_id="car_01", garage_id="gr_01", package_ids=["pk_wash", "pk_tire"], time="2026-11-20T10:00:00+07:00", pickup=True)], CS),
    T("carsvc_missing", MORN, "Đặt lịch bảo dưỡng xe ô tô ở Mazda Long Biên giúp anh.",
      [C("car_service.list_vehicles"), C("car_service.list_garages", query="Mazda Long Biên"), ASK("car_service.book_service", "time")], CS),
    # home_cleaning
    T("clean_book", MORN, "Thứ bảy này hai giờ chiều đặt tổng vệ sinh bốn tiếng.", [C("home_cleaning.get_packages"),
      C("home_cleaning.book", package_id="pkg_deep", start="2026-11-21T14:00:00+07:00", hours=4)], HC),
    T("clean_weekly", MORN, "Đặt dọn nhà mỗi tuần vào sáng chủ nhật tám giờ, mỗi lần ba tiếng, dặn chị giúp việc lau kỹ ban công.",
      [C("home_cleaning.get_packages"), C("home_cleaning.book", package_id="pkg_basic", start="2026-11-22T08:00:00+07:00", hours=3, repeat="weekly", note="lau kỹ ban công")], HC),
    T("clean_rate", MORN, "Buổi dọn nhà thứ bảy vừa rồi chị làm tốt lắm, đánh giá năm sao giúp anh.",
      [C("home_cleaning.list_bookings"), C("home_cleaning.rate", booking_id="hc_01", stars=5)], HC),
    T("clean_cancel", MORN, "Thứ bảy tuần sau nhà đi vắng, hủy lịch dọn nhà hôm đó nhé.", [C("home_cleaning.list_bookings"), C("home_cleaning.cancel", booking_id="hc_02")], HC),
    T("clean_missing_hours", MORN, "Đặt người giặt sofa sáng mai chín giờ.", [C("home_cleaning.get_packages"), ASK("home_cleaning.book", "hours")], HC),
    # traffic_fines
    T("fine_check", MORN, "Xe ô tô có bị phạt nguội không?", [C("traffic_fines.list_vehicles"), C("traffic_fines.check_fines", plate="30H-123.45")], F),
    T("fine_pay", MORN, "Có lỗi nào chưa nộp của xe ô tô thì nộp luôn bằng ví.",
      [C("traffic_fines.list_vehicles"), C("traffic_fines.check_fines", plate="30H-123.45"), C("traffic_fines.pay_fine", fine_id="fn_01", method="wallet")], F),
    T("fine_add", MORN, "Thêm biển số 51G-888.66 vào theo dõi, đặt tên là xe của em trai.", [C("traffic_fines.add_vehicle", plate="51G-888.66", nickname="xe của em trai")], F),
    T("fine_wife", MORN, "Kiểm tra xe máy của vợ có lỗi gì không.", [C("traffic_fines.list_vehicles"), C("traffic_fines.check_fines", plate="29B1-678.90")], F),
    # screen_time
    T("screen_limit", EVE, "Giới hạn Bông dùng điện thoại một tiếng mỗi ngày thôi.", [C("screen_time.get_report"), C("screen_time.set_daily_limit", kid_id="kid_02", minutes=60)], SC),
    T("screen_block", EVE, "Khóa YouTube trên iPad của Bi.", [C("screen_time.get_report"), C("screen_time.block_app", kid_id="kid_01", app_name="YouTube", blocked=True)], SC),
    T("screen_unblock", EVE, "Mở lại TikTok cho Bi đi.", [C("screen_time.get_report"), C("screen_time.block_app", kid_id="kid_01", app_name="TikTok", blocked=False)], SC),
    T("screen_bonus_cond", EVE, "Đứa nào dùng quá giới hạn hôm nay rồi thì thôi, còn đứa nào chưa thì cho thêm ba mươi phút.",
      [C("screen_time.get_report"), C("screen_time.grant_bonus", kid_id="kid_01", minutes=30)], SC),
    T("screen_missing", EVE, "Cho Bông thêm thời gian chơi nhé.", [C("screen_time.get_report"), ASK("screen_time.grant_bonus", "minutes")], SC),
    # photo_album
    T("photo_create", EVE, "Tạo album Chuyến đi Ba Vì gồm ảnh chụp từ thứ bảy tới chủ nhật vừa rồi.",
      [C("photo_album.create_album", title="Chuyến đi Ba Vì", date_from="2026-11-21", date_to="2026-11-22")], PH),
    T("photo_share", EVE, "Chia sẻ album sinh nhật bé Na cho mẹ.",
      [C("photo_album.list_albums", query="sinh nhật"), C("contacts.search", query="Mẹ"), C("photo_album.share_album", album_id="alb_01", contact_id="c_03")], PH),
    T("photo_delete", EVE, "Xóa album Sa Pa năm ngoái đi.", [C("photo_album.list_albums", query="Sa Pa"), C("photo_album.delete_album", album_id="alb_02")], PH),
    T("photo_missing", EVE, "Chia sẻ album du lịch Sa Pa nhé.", [C("photo_album.list_albums", query="Sa Pa"), ASK("photo_album.share_album", "contact_id")], PH),
    # charity
    T("charity_donate", MORN, "Ủng hộ năm trăm nghìn cho chiến dịch áo ấm vùng cao, để ẩn danh.",
      [C("charity.list_campaigns", query="áo ấm"), C("charity.donate", campaign_id="cp_01", amount=500000, anonymous=True)], CH),
    T("charity_monthly", MORN, "Mỗi tháng góp hai trăm nghìn cho bà con vùng lũ.",
      [C("charity.list_campaigns", query="lũ"), C("charity.donate", campaign_id="cp_02", amount=200000, monthly=True)], CH),
    T("charity_stop", MORN, "Dừng khoản ủng hộ hằng tháng cho bữa cơm 0 đồng giúp chị.", [C("charity.list_donations"), C("charity.stop_monthly", donation_id="don_01")], CH),
    T("charity_missing", MORN, "Ủng hộ chiến dịch bệnh viện K giúp anh.", [C("charity.list_campaigns", query="bệnh viện K"), ASK("charity.donate", "amount")], CH),
    # streaming
    T("stream_add_kids", EVE, "Thêm phim Doraemon vùng đất lý tưởng vào danh sách xem sau của hồ sơ trẻ em.",
      [C("streaming.search_titles", query="Doraemon"), C("streaming.add_to_watchlist", title_id="tt_03", profile="kids")], ST),
    T("stream_download_ep", EVE, "Tải tập ba của Hành Tinh Xanh chất lượng cao để mai đi máy bay xem.",
      [C("streaming.search_titles", query="Hành Tinh Xanh"), C("streaming.download", title_id="tt_04", episode=3, quality="high")], ST),
    T("stream_missing_ep", EVE, "Tải phim Gia Đình Là Số Một về máy.", [C("streaming.search_titles", query="Gia Đình Là Số Một"), ASK("streaming.download", "episode")], ST),
    T("stream_remove", EVE, "Xem Mắt Biếc rồi, bỏ nó khỏi danh sách xem sau.", [C("streaming.get_watchlist"), C("streaming.remove_from_watchlist", title_id="tt_01")], ST),
    T("stream_out_of_scope", EVE, "Mua vé xem phim Mắt Biếc ở rạp tối nay.", [], ST),
    # health_monitor
    T("health_bp", MORN, "Ghi huyết áp của bố sáng nay một trăm ba mươi tám trên tám mươi lăm.",
      [C("health_monitor.list_members"), C("health_monitor.log_reading", member_id="mb_01", type="blood_pressure", value=138, value2=85)], HM),
    T("health_glucose", MORN, "Đường huyết của mẹ lúc đói là sáu phẩy một, ghi lại giúp con.",
      [C("health_monitor.list_members"), C("health_monitor.log_reading", member_id="mb_02", type="glucose", value=6.1, note="lúc đói")], HM),
    T("health_weight", MORN, "Hôm nay mình cân được sáu mươi tám ký rưỡi.", [C("health_monitor.list_members"), C("health_monitor.log_reading", member_id="mb_03", type="weight", value=68.5)], HM),
    T("health_share", MORN, "Gửi chỉ số huyết áp hai tuần gần đây của bố cho bác sĩ qua email bs.hoa@phongkham.vn.",
      [C("health_monitor.list_members"), C("health_monitor.share_report", member_id="mb_01", email="bs.hoa@phongkham.vn", days=14)], HM),
    T("health_missing", MORN, "Ghi nhịp tim của mẹ giúp con.", [C("health_monitor.list_members"), ASK("health_monitor.log_reading", "value")], HM),
    # public_services
    T("gov_status", MORN, "Hồ sơ hộ chiếu của tôi tới đâu rồi?", [C("public_services.list_applications")], GV),
    T("gov_pay", MORN, "Nộp lệ phí hồ sơ hộ chiếu luôn đi.", [C("public_services.list_applications"), C("public_services.pay_fee", application_id="app_01")], GV),
    T("gov_book", MORN, "Đặt lịch làm căn cước ở công an phường hai giờ chiều nay, xong thêm vào lịch của tôi.",
      [C("public_services.list_offices", service="id_card"), C("public_services.book_slot", office_id="off_02", service="id_card", time="2026-11-19T14:00:00+07:00"),
       C("calendar.create_event", title="Làm căn cước ở công an phường Dịch Vọng", start="2026-11-19T14:00:00+07:00", location="Công an phường Dịch Vọng")], GVC),
    T("gov_missing", MORN, "Đặt lịch hẹn làm hộ chiếu cho tôi.", [C("public_services.list_offices", service="passport"), ASK("public_services.book_slot", "time")], GV),
    # car_remote
    T("car_precool", SAT, "Bật điều hòa xe hai mươi bốn độ trước mười lăm phút, chị xuống liền.", [C("car_remote.climate_start", temperature=24, minutes=15)], CR),
    T("car_lock_check", SAT, "Xe khóa cửa chưa, chưa thì khóa lại.", [C("car_remote.get_status"), C("car_remote.lock", locked=True)], CR),
    T("car_find", SAT, "Không nhớ để xe chỗ nào, nháy đèn thôi đừng bấm còi nha.", [C("car_remote.honk_flash", mode="flash")], CR),
    T("car_missing_temp", SAT, "Bật máy lạnh xe trước giùm anh.", [ASK("car_remote.climate_start", "temperature")], CR),
    T("car_hot", SAT, "Trong xe nóng trên ba mươi độ thì bật điều hòa hai mươi lăm độ.", [C("car_remote.get_status"), C("car_remote.climate_start", temperature=25)], CR),
    # baby_tracker
    T("baby_feed", EVE, "Su vừa bú bình một trăm năm mươi ml.", [C("baby_tracker.list_babies"), C("baby_tracker.log_feed", baby_id="bb_01", method="bottle", amount_ml=150)], BB),
    T("baby_sleep", EVE, "Ghi giấc ngủ của Su từ sáu rưỡi tới bảy giờ bốn mươi tối nay.",
      [C("baby_tracker.list_babies"), C("baby_tracker.log_sleep", baby_id="bb_01", start="2026-11-23T18:30:00+07:00", end="2026-11-23T19:40:00+07:00")], BB),
    T("baby_diaper", EVE, "Mới thay tã ướt cho bé.", [C("baby_tracker.list_babies"), C("baby_tracker.log_diaper", baby_id="bb_01", kind="wet")], BB),
    T("baby_last_feed", EVE, "Lần bú gần nhất của Su là mấy giờ?", [C("baby_tracker.list_babies"), C("baby_tracker.last_events", baby_id="bb_01", type="feed")], BB),
    T("baby_feed_missing", EVE, "Ghi lại cữ bú của Su.", [C("baby_tracker.list_babies"), ASK("baby_tracker.log_feed", "method")], BB),
    # savings_goals
    T("goal_deposit", SAT, "Góp thêm một triệu vào quỹ du lịch Phú Quốc.", [C("savings_goals.list_goals"), C("savings_goals.deposit", goal_id="goal_02", amount=1000000)], SG),
    T("goal_create", SAT, "Tạo mục tiêu tiết kiệm mua laptop hai mươi lăm triệu, hạn cuối tháng ba năm sau.",
      [C("savings_goals.create_goal", name="mua laptop", target=25000000, deadline="2027-03-31")], SG),
    T("goal_auto", SAT, "Mỗi tháng ngày mùng năm tự động góp hai triệu vào quỹ mua xe máy cho con.",
      [C("savings_goals.list_goals"), C("savings_goals.set_auto_deposit", goal_id="goal_01", amount=2000000, day_of_month=5)], SG),
    T("goal_withdraw_cond", SAT, "Quỹ Phú Quốc mà đủ mười lăm triệu rồi thì rút ra năm triệu, chưa đủ thì thôi.", [C("savings_goals.list_goals")], SG),
    T("goal_missing", SAT, "Rút tiền ở quỹ mua xe ra nhé.", [C("savings_goals.list_goals"), ASK("savings_goals.withdraw", "amount")], SG),
]
