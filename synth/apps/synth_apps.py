"""Synthetic apps (training only), in the zaic_tools App style: one class per app, one method per tool, never raises.
The builder copies this module into a copy of the TRAIN zaic_tools package (apps/), so the official Session runs it."""
from __future__ import annotations

from .base import App, read, write


def _day(ts):
    return str(ts or "")[:10]


class HotelBooking(App):
    name = "hotel_booking"

    @read
    def search_hotels(self, city=None, query=None, max_price=None, **_):
        out = self.read("hotels", eq={"city": city}, contains={("name", "address"): query})
        if max_price is not None:
            out["hotels"] = [h for h in out["hotels"] if h.get("price_per_night", 0) <= max_price]
        return out

    @read
    def list_bookings(self, status=None, **_):
        return self.read("bookings", eq={"status": status})

    @write
    def book(self, hotel_id=None, room_type=None, check_in=None, check_out=None, guests=None, note=None, **_):
        h = self.find("hotels", hotel_id)
        if h is None:
            return self.noop(f"không tìm thấy hotel_id={hotel_id!r}")
        if not (check_in and check_out and check_in < check_out):
            return self.noop("ngày trả phòng phải sau ngày nhận phòng")
        if h.get("rooms", {}).get(room_type, 0) <= 0:
            return self.noop(f"khách sạn {h['id']} hết phòng {room_type}")
        h["rooms"][room_type] -= 1
        r = self.create("bookings", "bk", hotel_id=h["id"], room_type=room_type, check_in=check_in, check_out=check_out, guests=guests,
                        note=note, status="confirmed")
        return {"ok": True, "booking": r["item"]}

    @write
    def change_dates(self, booking_id=None, check_in=None, check_out=None, **_):
        if not (check_in and check_out and check_in < check_out):
            return self.noop("ngày trả phòng phải sau ngày nhận phòng")
        return self.patch("bookings", "booking_id", booking_id, check_in=check_in, check_out=check_out)

    @write
    def cancel_booking(self, booking_id=None, **_):
        b = self.find("bookings", booking_id)
        if b is None or b.get("status") == "cancelled":
            return self.noop(f"không có đặt phòng đang hiệu lực booking_id={booking_id!r}")
        b["status"] = "cancelled"
        h = self.find("hotels", b.get("hotel_id"))
        if h is not None:
            h.setdefault("rooms", {})[b["room_type"]] = h["rooms"].get(b["room_type"], 0) + 1
        return {"ok": True, "booking": b}


class Clinic(App):
    name = "clinic"

    @read
    def search_doctors(self, specialty=None, query=None, **_):
        return self.read("doctors", eq={"specialty": specialty}, contains={("name", "clinic"): query})

    @read
    def list_appointments(self, **_):
        return self.read("appointments")

    def _take(self, doctor, time):
        slots = doctor.get("available_slots", [])
        hit = next((x for x in slots if x[:16] == str(time)[:16]), None)
        if hit is None:
            return False
        slots.remove(hit)
        return True

    @write
    def book_appointment(self, doctor_id=None, time=None, patient=None, reason=None, **_):
        d = self.find("doctors", doctor_id)
        if d is None:
            return self.noop(f"không tìm thấy doctor_id={doctor_id!r}")
        if not self._take(d, time):
            return self.noop(f"bác sĩ {d['id']} không còn trống lúc {time}")
        r = self.create("appointments", "apt", doctor_id=d["id"], time=time, patient=patient or "self", reason=reason, status="booked")
        return {"ok": True, "appointment": r["item"]}

    @write
    def reschedule(self, appointment_id=None, time=None, **_):
        a = self.find("appointments", appointment_id)
        if a is None or a.get("status") != "booked":
            return self.noop(f"không có lịch khám appointment_id={appointment_id!r}")
        d = self.find("doctors", a["doctor_id"])
        if d is None or not self._take(d, time):
            return self.noop(f"bác sĩ không còn trống lúc {time}")
        d.setdefault("available_slots", []).append(a["time"][:16])
        a["time"] = time
        return {"ok": True, "appointment": a}

    @write
    def cancel_appointment(self, appointment_id=None, **_):
        a = self.find("appointments", appointment_id)
        if a is None or a.get("status") != "booked":
            return self.noop(f"không có lịch khám appointment_id={appointment_id!r}")
        a["status"] = "cancelled"
        return {"ok": True, "appointment": a}


class PetCare(App):
    name = "pet_care"

    @read
    def list_pets(self, **_):
        return self.read("pets")

    @read
    def get_logs(self, pet_id=None, type=None, **_):
        return self.read("logs", eq={"pet_id": pet_id, "type": type})

    @write
    def log_activity(self, pet_id=None, type=None, amount=None, note=None, **_):
        if self.find("pets", pet_id) is None:
            return self.noop(f"không tìm thấy pet_id={pet_id!r}")
        r = self.create("logs", "log", pet_id=pet_id, type=type, time=self.session.now, amount=amount, note=note)
        return {"ok": True, "log": r["item"]}

    @write
    def schedule_care(self, pet_id=None, type=None, time=None, repeat=None, **_):
        if self.find("pets", pet_id) is None:
            return self.noop(f"không tìm thấy pet_id={pet_id!r}")
        r = self.create("schedules", "sch", pet_id=pet_id, type=type, time=time, repeat=repeat or "none")
        return {"ok": True, "schedule": r["item"]}

    @write
    def update_weight(self, pet_id=None, weight_kg=None, **_):
        return self.patch("pets", "pet_id", pet_id, weight_kg=weight_kg)


class Laundry(App):
    name = "laundry"

    @read
    def list_services(self, **_):
        return self.read("services")

    @read
    def list_orders(self, status=None, **_):
        return self.read("orders", eq={"status": status})

    @write
    def create_order(self, items=None, pickup_time=None, address=None, express=None, **_):
        lines, total = [], 0
        for it in items or []:
            sv = self.find("services", it.get("service_id"))
            if sv is None:
                return self.noop(f"không tìm thấy service_id={it.get('service_id')!r}")
            q = it.get("quantity")
            lines.append({"service_id": sv["id"], "quantity": q})
            total += sv["price"] * q
        if not lines:
            return self.noop("đơn không có dịch vụ nào")
        total = int(total * (1.5 if express else 1))
        r = self.create("orders", "ord", items=lines, pickup_time=pickup_time, address=address or self.state.get("default_address"),
                        express=bool(express), total=total, status="scheduled")
        return {"ok": True, "order": r["item"]}

    @write
    def cancel_order(self, order_id=None, **_):
        o = self.find("orders", order_id)
        if o is None or o.get("status") != "scheduled":
            return self.noop(f"đơn {order_id!r} không huỷ được")
        o["status"] = "cancelled"
        return {"ok": True, "order": o}


class Salon(App):
    name = "salon"

    @read
    def search_salons(self, query=None, **_):
        return self.read("salons", contains={("name", "address"): query}, drop=("services", "stylists"))

    @read
    def get_salon(self, salon_id=None, **_):
        sl = self.find("salons", salon_id)
        if sl is None:
            return self.noop(f"không tìm thấy salon_id={salon_id!r}")
        return {"ok": True, "salon": sl}

    @read
    def list_bookings(self, **_):
        return self.read("bookings")

    @write
    def book(self, salon_id=None, service_ids=None, time=None, stylist_id=None, **_):
        sl = self.find("salons", salon_id)
        if sl is None:
            return self.noop(f"không tìm thấy salon_id={salon_id!r}")
        known = {x["id"]: x for x in sl.get("services", [])}
        bad = [x for x in service_ids or [] if x not in known]
        if bad or not service_ids:
            return self.noop(f"dịch vụ không có ở salon {sl['id']}: {bad}")
        if stylist_id and stylist_id not in {x["id"] for x in sl.get("stylists", [])}:
            return self.noop(f"không có thợ stylist_id={stylist_id!r}")
        total = sum(known[x]["price"] for x in service_ids)
        r = self.create("bookings", "sb", salon_id=sl["id"], service_ids=list(service_ids), time=time, stylist_id=stylist_id, total=total, status="booked")
        return {"ok": True, "booking": r["item"]}

    @write
    def cancel_booking(self, booking_id=None, **_):
        bk = self.find("bookings", booking_id)
        if bk is None or bk.get("status") != "booked":
            return self.noop(f"không có lịch hẹn booking_id={booking_id!r}")
        bk["status"] = "cancelled"
        return {"ok": True, "booking": bk}


class EvCharging(App):
    name = "ev_charging"

    @read
    def get_vehicle(self, **_):
        return {"ok": True, "vehicle": self.state.get("vehicle", {})}

    @read
    def find_stations(self, connector=None, min_power_kw=None, query=None, **_):
        out = self.read("stations", contains={("name", "address"): query})
        out["stations"] = [x for x in out["stations"] if (not connector or connector in x.get("connectors", []))
                           and (min_power_kw is None or x.get("power_kw", 0) >= min_power_kw)]
        return out

    @read
    def list_sessions(self, **_):
        return self.read("sessions")

    @write
    def reserve(self, station_id=None, start=None, duration_min=None, target_percent=None, **_):
        st = self.find("stations", station_id)
        if st is None:
            return self.noop(f"không tìm thấy station_id={station_id!r}")
        if st.get("free_ports", 0) <= 0:
            return self.noop(f"trạm {st['id']} hết cổng trống")
        st["free_ports"] -= 1
        r = self.create("sessions", "chg", station_id=st["id"], start=start, duration_min=duration_min, target_percent=target_percent, status="reserved")
        return {"ok": True, "session": r["item"]}

    @write
    def cancel_session(self, session_id=None, **_):
        x = self.find("sessions", session_id)
        if x is None or x.get("status") != "reserved":
            return self.noop(f"không có lượt sạc session_id={session_id!r}")
        x["status"] = "cancelled"
        st = self.find("stations", x["station_id"])
        if st is not None:
            st["free_ports"] = st.get("free_ports", 0) + 1
        return {"ok": True, "session": x}


class WifiRouter(App):
    name = "wifi_router"

    @read
    def get_status(self, **_):
        net = {k: v for k, v in self.state.get("network", {}).items() if k not in ("password", "guest_password")}
        return {"ok": True, "network": net, "devices": self.read("devices")["devices"], "schedules": self.read("schedules")["schedules"]}

    @write
    def block_device(self, device_id=None, blocked=None, **_):
        return self.patch("devices", "device_id", device_id, set={"blocked": bool(blocked)})

    @write
    def set_guest_network(self, enabled=None, password=None, **_):
        net = self.state.setdefault("network", {})
        net["guest_enabled"] = bool(enabled)
        if password:
            net["guest_password"] = password
        return {"ok": True, "guest_enabled": net["guest_enabled"]}

    @write
    def set_schedule(self, device_id=None, off_from=None, off_to=None, days=None, **_):
        if self.find("devices", device_id) is None:
            return self.noop(f"không tìm thấy device_id={device_id!r}")
        r = self.create("schedules", "ws", device_id=device_id, off_from=off_from, off_to=off_to, days=sorted(days) if days else ["all"])
        return {"ok": True, "schedule": r["item"]}

    @write
    def change_password(self, new_password=None, **_):
        self.state.setdefault("network", {})["password"] = new_password
        return {"ok": True}


class ExpenseTracker(App):
    name = "expense_tracker"

    def _spent(self, cid):
        month = _day(self.session.now)[:7]
        return sum(e["amount"] for e in self.state.get("expenses", []) if e.get("category_id") == cid and str(e.get("date", ""))[:7] == month)

    @read
    def list_categories(self, **_):
        cats = self.read("categories")["categories"]
        return {"ok": True, "categories": [{**c, "spent": self._spent(c["id"])} for c in cats]}

    @read
    def list_expenses(self, category_id=None, date_from=None, date_to=None, **_):
        return self.read("expenses", eq={"category_id": category_id}, date_range=("date", date_from, date_to))

    @write
    def add_expense(self, amount=None, category_id=None, date=None, note=None, **_):
        if self.find("categories", category_id) is None:
            return self.noop(f"không tìm thấy category_id={category_id!r}")
        r = self.create("expenses", "exp", category_id=category_id, amount=amount, date=date or _day(self.session.now), note=note)
        return {"ok": True, "expense": r["item"]}

    @write
    def set_budget(self, category_id=None, budget=None, **_):
        return self.patch("categories", "category_id", category_id, budget=budget)

    @write
    def delete_expense(self, expense_id=None, **_):
        return self.delete("expenses", "expense_id", expense_id)


class SchoolPortal(App):
    name = "school_portal"

    @read
    def list_children(self, **_):
        return self.read("children")

    @read
    def get_grades(self, child_id=None, subject=None, **_):
        return self.read("grades", where={"child_id": child_id}, eq={"subject": subject})

    @read
    def get_notices(self, child_id=None, **_):
        out = self.read("notices")
        if child_id:
            c = self.find("children", child_id)
            out["notices"] = [x for x in out["notices"] if x.get("class") in (None, "all", (c or {}).get("class"))]
        return out

    @write
    def request_leave(self, child_id=None, date=None, days=None, reason=None, **_):
        if self.find("children", child_id) is None:
            return self.noop(f"không tìm thấy child_id={child_id!r}")
        r = self.create("leave_requests", "lv", child_id=child_id, date=date, days=days or 1, reason=reason, status="sent")
        return {"ok": True, "leave_request": r["item"]}

    @write
    def message_teacher(self, child_id=None, content=None, **_):
        c = self.find("children", child_id)
        if c is None:
            return self.noop(f"không tìm thấy child_id={child_id!r}")
        r = self.create("messages", "msg", child_id=child_id, teacher=c.get("teacher"), content=content, time=self.session.now)
        return {"ok": True, "message": r["item"]}


class Apartment(App):
    name = "apartment"

    @read
    def list_fees(self, paid=None, **_):
        out = self.read("fees")
        if paid is not None:
            out["fees"] = [f for f in out["fees"] if bool(f.get("paid")) == bool(paid)]
        return out

    @write
    def pay_fee(self, fee_id=None, **_):
        f = self.find("fees", fee_id)
        if f is None or f.get("paid"):
            return self.noop(f"khoản phí {fee_id!r} không có hoặc đã đóng")
        f["paid"] = True
        return {"ok": True, "fee": f}

    @write
    def create_request(self, category=None, description=None, preferred_time=None, **_):
        r = self.create("requests", "req", category=category, description=description, preferred_time=preferred_time, status="open")
        return {"ok": True, "request": r["item"]}

    @read
    def list_requests(self, **_):
        return self.read("requests")

    @write
    def register_guest(self, name=None, arrival=None, vehicle_plate=None, **_):
        r = self.create("guests", "gst", name=name, arrival=arrival, vehicle_plate=vehicle_plate)
        return {"ok": True, "guest": r["item"]}


class SportsCourt(App):
    name = "sports_court"

    @read
    def search_courts(self, sport=None, date=None, **_):
        out = []
        for c in self.state.get("courts", []):
            if c.get("sport") == sport:
                out.append({k: v for k, v in c.items() if k != "slots"} | {"date": date, "free_slots": list(c.get("slots", {}).get(date, []))})
        return {"ok": True, "courts": out}

    @write
    def book_court(self, court_id=None, date=None, start_time=None, hours=None, **_):
        c = self.find("courts", court_id)
        if c is None:
            return self.noop(f"không tìm thấy court_id={court_id!r}")
        free = c.setdefault("slots", {}).get(date, [])
        h0 = int(str(start_time)[:2]) if str(start_time)[:2].isdigit() else -1
        need = [f"{h0 + k:02d}:00" for k in range(hours or 1)]
        if str(start_time)[3:5] != "00" or any(x not in free for x in need):
            return self.noop(f"sân {c['id']} không trống từ {start_time} trong {hours} giờ")
        for x in need:
            free.remove(x)
        r = self.create("bookings", "cb", court_id=c["id"], date=date, start_time=start_time, hours=hours,
                        total=c.get("price_per_hour", 0) * (hours or 1), status="booked")
        return {"ok": True, "booking": r["item"]}

    @read
    def list_bookings(self, **_):
        return self.read("bookings")

    @write
    def cancel_booking(self, booking_id=None, **_):
        bk = self.find("bookings", booking_id)
        if bk is None or bk.get("status") != "booked":
            return self.noop(f"không có lượt đặt booking_id={booking_id!r}")
        bk["status"] = "cancelled"
        return {"ok": True, "booking": bk}


class Insurance(App):
    name = "insurance"

    @read
    def list_policies(self, type=None, **_):
        return self.read("policies", eq={"type": type})

    @write
    def renew_policy(self, policy_id=None, months=None, **_):
        p = self.find("policies", policy_id)
        if p is None:
            return self.noop(f"không tìm thấy policy_id={policy_id!r}")
        m = months or 12
        y, mo, d = (int(x) for x in p["end_date"].split("-"))
        mo += m
        y, mo = y + (mo - 1) // 12, (mo - 1) % 12 + 1
        p["end_date"] = f"{y:04d}-{mo:02d}-{d:02d}"
        p["status"] = "active"
        return {"ok": True, "policy": p}

    @write
    def file_claim(self, policy_id=None, incident_date=None, amount=None, description=None, **_):
        p = self.find("policies", policy_id)
        if p is None or p.get("status") != "active":
            return self.noop(f"hợp đồng {policy_id!r} không còn hiệu lực")
        r = self.create("claims", "clm", policy_id=p["id"], incident_date=incident_date, amount=amount, description=description, status="submitted")
        return {"ok": True, "claim": r["item"]}

    @read
    def list_claims(self, **_):
        return self.read("claims")


class CloudDrive(App):
    name = "cloud_drive"

    @read
    def search_files(self, query=None, folder=None, **_):
        out = self.read("files", contains={("name",): query}, eq={"folder": folder})
        out["files"] = [f for f in out["files"] if not f.get("trashed")]
        return out

    @write
    def share_file(self, file_id=None, contact_id=None, email=None, permission=None, **_):
        f = self.find("files", file_id)
        if f is None or f.get("trashed"):
            return self.noop(f"không tìm thấy file_id={file_id!r}")
        who = contact_id or email
        f.setdefault("shared_with", [])
        f["shared_with"] = [x for x in f["shared_with"] if x.get("who") != who] + [{"who": who, "permission": permission or "view"}]
        return {"ok": True, "file": f}

    @write
    def move_file(self, file_id=None, folder=None, **_):
        return self.patch("files", "file_id", file_id, folder=folder)

    @write
    def trash_file(self, file_id=None, **_):
        return self.patch("files", "file_id", file_id, set={"trashed": True})


class SmartGarden(App):
    name = "smart_garden"

    @read
    def get_status(self, **_):
        return {"ok": True, "zones": self.read("zones")["zones"], "schedules": self.read("schedules")["schedules"]}

    @write
    def water_now(self, zone_id=None, duration_min=None, **_):
        z = self.find("zones", zone_id)
        if z is None:
            return self.noop(f"không tìm thấy zone_id={zone_id!r}")
        z["last_watered"] = self.session.now
        z["moisture"] = min(100, z.get("moisture", 0) + 2 * (duration_min or 0))
        return {"ok": True, "zone": z}

    @write
    def set_schedule(self, zone_id=None, time=None, duration_min=None, days=None, **_):
        if self.find("zones", zone_id) is None:
            return self.noop(f"không tìm thấy zone_id={zone_id!r}")
        r = self.create("schedules", "gs", zone_id=zone_id, time=time, duration_min=duration_min, days=sorted(days) if days else ["all"], paused_until=None)
        return {"ok": True, "schedule": r["item"]}

    @write
    def pause_schedule(self, schedule_id=None, until=None, **_):
        return self.patch("schedules", "schedule_id", schedule_id, paused_until=until)


# ---- batch 2 -------------------------------------------------------------------------------------------------------------
class RestaurantBooking(App):
    name = "restaurant_booking"

    @read
    def search(self, query=None, **_):
        return self.read("restaurants", contains={("name", "cuisine", "address"): query})

    @read
    def list_reservations(self, **_):
        return self.read("reservations")

    def _take(self, r, time):
        hit = next((x for x in r.get("free_times", []) if x[:16] == str(time)[:16]), None)
        if hit is not None:
            r["free_times"].remove(hit)
        return hit is not None

    @write
    def reserve(self, restaurant_id=None, time=None, party_size=None, note=None, **_):
        r = self.find("restaurants", restaurant_id)
        if r is None:
            return self.noop(f"không tìm thấy restaurant_id={restaurant_id!r}")
        if party_size and party_size > r.get("max_party", 12):
            return self.noop(f"nhà hàng {r['id']} nhận tối đa {r.get('max_party', 12)} người")
        if not self._take(r, time):
            return self.noop(f"nhà hàng {r['id']} hết bàn lúc {time}")
        x = self.create("reservations", "rsv", restaurant_id=r["id"], time=time, party_size=party_size, note=note, status="booked")
        return {"ok": True, "reservation": x["item"]}

    @write
    def change_reservation(self, reservation_id=None, time=None, party_size=None, **_):
        x = self.find("reservations", reservation_id)
        if x is None or x.get("status") != "booked":
            return self.noop(f"không có bàn đặt reservation_id={reservation_id!r}")
        if time:
            r = self.find("restaurants", x["restaurant_id"])
            if r is None or not self._take(r, time):
                return self.noop(f"hết bàn lúc {time}")
            r.setdefault("free_times", []).append(x["time"][:16])
            x["time"] = time
        if party_size:
            x["party_size"] = party_size
        return {"ok": True, "reservation": x}

    @write
    def cancel_reservation(self, reservation_id=None, **_):
        x = self.find("reservations", reservation_id)
        if x is None or x.get("status") != "booked":
            return self.noop(f"không có bàn đặt reservation_id={reservation_id!r}")
        x["status"] = "cancelled"
        return {"ok": True, "reservation": x}


class CarService(App):
    name = "car_service"

    @read
    def list_vehicles(self, **_):
        return self.read("vehicles")

    @read
    def list_garages(self, query=None, **_):
        return self.read("garages", contains={("name", "address"): query})

    @read
    def list_appointments(self, **_):
        return self.read("appointments")

    @write
    def book_service(self, vehicle_id=None, garage_id=None, package_ids=None, time=None, pickup=None, **_):
        v, g = self.find("vehicles", vehicle_id), self.find("garages", garage_id)
        if v is None or g is None:
            return self.noop("không tìm thấy xe hoặc garage")
        known = {p["id"]: p for p in g.get("packages", [])}
        if not package_ids or any(p not in known for p in package_ids):
            return self.noop(f"gói dịch vụ không có ở garage {g['id']}")
        x = self.create("appointments", "svc", vehicle_id=v["id"], garage_id=g["id"], package_ids=list(package_ids), time=time, pickup=bool(pickup),
                        total=sum(known[p]["price"] for p in package_ids), status="booked")
        return {"ok": True, "appointment": x["item"]}

    @write
    def cancel_appointment(self, appointment_id=None, **_):
        x = self.find("appointments", appointment_id)
        if x is None or x.get("status") != "booked":
            return self.noop(f"không có lịch appointment_id={appointment_id!r}")
        x["status"] = "cancelled"
        return {"ok": True, "appointment": x}


class HomeCleaning(App):
    name = "home_cleaning"

    @read
    def get_packages(self, **_):
        return self.read("packages")

    @read
    def list_bookings(self, **_):
        return self.read("bookings")

    @write
    def book(self, package_id=None, start=None, hours=None, repeat=None, note=None, **_):
        p = self.find("packages", package_id)
        if p is None:
            return self.noop(f"không tìm thấy package_id={package_id!r}")
        x = self.create("bookings", "hc", package_id=p["id"], start=start, hours=hours, repeat=repeat or "none", note=note,
                        total=p["price_per_hour"] * hours, status="booked", rating=None)
        return {"ok": True, "booking": x["item"]}

    @write
    def cancel(self, booking_id=None, **_):
        x = self.find("bookings", booking_id)
        if x is None or x.get("status") != "booked":
            return self.noop(f"không có lịch booking_id={booking_id!r}")
        x["status"] = "cancelled"
        return {"ok": True, "booking": x}

    @write
    def rate(self, booking_id=None, stars=None, comment=None, **_):
        x = self.find("bookings", booking_id)
        if x is None or x.get("status") != "done":
            return self.noop("chỉ đánh giá được buổi đã xong")
        x["rating"] = stars
        x["comment"] = comment
        return {"ok": True, "booking": x}


class TrafficFines(App):
    name = "traffic_fines"

    @read
    def list_vehicles(self, **_):
        return self.read("vehicles")

    @read
    def check_fines(self, plate=None, **_):
        return self.read("fines", eq={"plate": plate})

    @write
    def pay_fine(self, fine_id=None, method=None, **_):
        f = self.find("fines", fine_id)
        if f is None or f.get("paid"):
            return self.noop(f"vi phạm {fine_id!r} không có hoặc đã nộp")
        f["paid"] = True
        f["method"] = method or "linked_card"
        return {"ok": True, "fine": f}

    @write
    def add_vehicle(self, plate=None, nickname=None, **_):
        if any(v.get("plate") == plate for v in self.state.get("vehicles", [])):
            return self.noop(f"biển số {plate} đã có")
        x = self.create("vehicles", "veh", plate=plate, nickname=nickname)
        return {"ok": True, "vehicle": x["item"]}


class ScreenTime(App):
    name = "screen_time"

    @read
    def get_report(self, **_):
        return self.read("kids")

    @write
    def set_daily_limit(self, kid_id=None, minutes=None, **_):
        return self.patch("kids", "kid_id", kid_id, set={"daily_limit_min": minutes})

    @write
    def block_app(self, kid_id=None, app_name=None, blocked=None, **_):
        k = self.find("kids", kid_id)
        if k is None:
            return self.noop(f"không tìm thấy kid_id={kid_id!r}")
        apps = set(k.get("blocked_apps", []))
        apps = apps | {app_name} if blocked else apps - {app_name}
        k["blocked_apps"] = sorted(apps)
        return {"ok": True, "kid": k}

    @write
    def grant_bonus(self, kid_id=None, minutes=None, **_):
        k = self.find("kids", kid_id)
        if k is None:
            return self.noop(f"không tìm thấy kid_id={kid_id!r}")
        k["bonus_today_min"] = k.get("bonus_today_min", 0) + minutes
        return {"ok": True, "kid": k}


class PhotoAlbum(App):
    name = "photo_album"

    @read
    def list_albums(self, query=None, **_):
        return self.read("albums", contains={("title",): query})

    @write
    def create_album(self, title=None, date_from=None, date_to=None, **_):
        photos = [p["id"] for p in self.state.get("photos", []) if date_from and str(p.get("date", "")) >= date_from and str(p.get("date", "")) <= (date_to or date_from)]
        x = self.create("albums", "alb", title=title, photo_ids=photos, shared_with=[])
        return {"ok": True, "album": x["item"]}

    @write
    def share_album(self, album_id=None, contact_id=None, **_):
        a = self.find("albums", album_id)
        if a is None:
            return self.noop(f"không tìm thấy album_id={album_id!r}")
        a["shared_with"] = sorted(set(a.get("shared_with", [])) | {contact_id})
        return {"ok": True, "album": a}

    @write
    def delete_album(self, album_id=None, **_):
        return self.delete("albums", "album_id", album_id)


class Charity(App):
    name = "charity"

    @read
    def list_campaigns(self, query=None, **_):
        return self.read("campaigns", contains={("title", "topic", "organizer"): query})

    @read
    def list_donations(self, **_):
        return self.read("donations")

    @write
    def donate(self, campaign_id=None, amount=None, monthly=None, anonymous=None, **_):
        c = self.find("campaigns", campaign_id)
        if c is None:
            return self.noop(f"không tìm thấy campaign_id={campaign_id!r}")
        c["raised"] = c.get("raised", 0) + amount
        x = self.create("donations", "don", campaign_id=c["id"], amount=amount, monthly=bool(monthly), anonymous=bool(anonymous), active=True)
        return {"ok": True, "donation": x["item"]}

    @write
    def stop_monthly(self, donation_id=None, **_):
        d = self.find("donations", donation_id)
        if d is None or not d.get("monthly") or not d.get("active"):
            return self.noop("không có khoản ủng hộ hằng tháng đang chạy")
        d["active"] = False
        return {"ok": True, "donation": d}


class Streaming(App):
    name = "streaming"

    @read
    def search_titles(self, query=None, **_):
        return self.read("titles", contains={("name", "genre"): query})

    @read
    def get_watchlist(self, profile=None, **_):
        ids = self.state.get("watchlist", {}).get(profile or "main", [])
        return {"ok": True, "watchlist": [t for t in self.state.get("titles", []) if t["id"] in ids]}

    @write
    def add_to_watchlist(self, title_id=None, profile=None, **_):
        if self.find("titles", title_id) is None:
            return self.noop(f"không tìm thấy title_id={title_id!r}")
        wl = self.state.setdefault("watchlist", {}).setdefault(profile or "main", [])
        if title_id in wl:
            return self.noop("đã có trong danh sách")
        wl.append(title_id)
        return {"ok": True, "watchlist": wl}

    @write
    def remove_from_watchlist(self, title_id=None, profile=None, **_):
        wl = self.state.setdefault("watchlist", {}).setdefault(profile or "main", [])
        if title_id not in wl:
            return self.noop("không có trong danh sách")
        wl.remove(title_id)
        return {"ok": True, "watchlist": wl}

    @write
    def download(self, title_id=None, episode=None, quality=None, **_):
        t = self.find("titles", title_id)
        if t is None:
            return self.noop(f"không tìm thấy title_id={title_id!r}")
        if t.get("type") == "series" and not episode:
            return self.noop("phim bộ cần số tập")
        x = self.create("downloads", "dl", title_id=t["id"], episode=episode, quality=quality or "standard")
        return {"ok": True, "download": x["item"]}


class HealthMonitor(App):
    name = "health_monitor"

    @read
    def list_members(self, **_):
        return self.read("members")

    @read
    def get_history(self, member_id=None, type=None, **_):
        return self.read("readings", where={"member_id": member_id}, eq={"type": type})

    @write
    def log_reading(self, member_id=None, type=None, value=None, value2=None, note=None, **_):
        if self.find("members", member_id) is None:
            return self.noop(f"không tìm thấy member_id={member_id!r}")
        x = self.create("readings", "rd", member_id=member_id, type=type, value=value, value2=value2, note=note, time=self.session.now)
        return {"ok": True, "reading": x["item"]}

    @write
    def share_report(self, member_id=None, email=None, days=None, **_):
        if self.find("members", member_id) is None:
            return self.noop(f"không tìm thấy member_id={member_id!r}")
        x = self.create("shares", "shr", member_id=member_id, email=email, days=days or 30)
        return {"ok": True, "share": x["item"]}


class PublicServices(App):
    name = "public_services"

    @read
    def list_applications(self, **_):
        return self.read("applications")

    @read
    def list_offices(self, service=None, **_):
        out = [o for o in self.state.get("offices", []) if service in o.get("services", [])]
        return {"ok": True, "offices": out}

    @write
    def book_slot(self, office_id=None, service=None, time=None, **_):
        o = self.find("offices", office_id)
        if o is None or service not in o.get("services", []):
            return self.noop("cơ quan không nhận dịch vụ này")
        hit = next((x for x in o.get("free_slots", []) if x[:16] == str(time)[:16]), None)
        if hit is None:
            return self.noop(f"không còn lịch lúc {time}")
        o["free_slots"].remove(hit)
        x = self.create("appointments", "gov", office_id=o["id"], service=service, time=time)
        return {"ok": True, "appointment": x["item"]}

    @write
    def pay_fee(self, application_id=None, **_):
        a = self.find("applications", application_id)
        if a is None or a.get("fee_paid") or not a.get("fee"):
            return self.noop("hồ sơ không có lệ phí cần nộp")
        a["fee_paid"] = True
        return {"ok": True, "application": a}


class CarRemote(App):
    name = "car_remote"

    @read
    def get_status(self, **_):
        return {"ok": True, "car": self.state.get("car", {})}

    @write
    def lock(self, locked=None, **_):
        self.state.setdefault("car", {})["locked"] = bool(locked)
        return {"ok": True, "locked": bool(locked)}

    @write
    def climate_start(self, temperature=None, minutes=None, **_):
        c = self.state.setdefault("car", {})
        c["climate"] = {"on": True, "temperature": temperature, "minutes": minutes or 10}
        return {"ok": True, "climate": c["climate"]}

    @write
    def climate_stop(self, **_):
        c = self.state.setdefault("car", {})
        if not (c.get("climate") or {}).get("on"):
            return self.noop("điều hoà đang tắt")
        c["climate"] = {"on": False}
        return {"ok": True}

    @write
    def honk_flash(self, mode=None, **_):
        self.state.setdefault("car", {})["last_locate"] = {"mode": mode, "time": self.session.now}
        return {"ok": True}


class BabyTracker(App):
    name = "baby_tracker"

    @read
    def list_babies(self, **_):
        return self.read("babies")

    @read
    def last_events(self, baby_id=None, type=None, **_):
        out = self.read("events", where={"baby_id": baby_id}, eq={"type": type})
        out["events"] = sorted(out["events"], key=lambda e: e.get("time", ""), reverse=True)[:5]
        return out

    def _log(self, baby_id, **fields):
        if self.find("babies", baby_id) is None:
            return self.noop(f"không tìm thấy baby_id={baby_id!r}")
        x = self.create("events", "ev", baby_id=baby_id, **fields)
        return {"ok": True, "event": x["item"]}

    @write
    def log_feed(self, baby_id=None, method=None, amount_ml=None, time=None, **_):
        return self._log(baby_id, type="feed", method=method, amount_ml=amount_ml, time=time or self.session.now)

    @write
    def log_sleep(self, baby_id=None, start=None, end=None, **_):
        return self._log(baby_id, type="sleep", time=start, end=end)

    @write
    def log_diaper(self, baby_id=None, kind=None, **_):
        return self._log(baby_id, type="diaper", kind=kind, time=self.session.now)


class SavingsGoals(App):
    name = "savings_goals"

    @read
    def list_goals(self, **_):
        return self.read("goals")

    @write
    def create_goal(self, name=None, target=None, deadline=None, **_):
        x = self.create("goals", "goal", name=name, target=target, deadline=deadline, saved=0, auto=None)
        return {"ok": True, "goal": x["item"]}

    @write
    def deposit(self, goal_id=None, amount=None, **_):
        g = self.find("goals", goal_id)
        if g is None:
            return self.noop(f"không tìm thấy goal_id={goal_id!r}")
        g["saved"] = g.get("saved", 0) + amount
        return {"ok": True, "goal": g}

    @write
    def withdraw(self, goal_id=None, amount=None, **_):
        g = self.find("goals", goal_id)
        if g is None or g.get("saved", 0) < amount:
            return self.noop("không đủ tiền trong mục tiêu")
        g["saved"] -= amount
        return {"ok": True, "goal": g}

    @write
    def set_auto_deposit(self, goal_id=None, amount=None, day_of_month=None, **_):
        return self.patch("goals", "goal_id", goal_id, set={"auto": {"amount": amount, "day_of_month": day_of_month}})
