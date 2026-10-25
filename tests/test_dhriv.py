import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from dhriv import (
    AutoCompiler,
    DhrivInterpreter,
    DhrivRuntimeError,
    DhrivSyntaxError,
    SelfLearningEngine,
    WatchDog,
    WatchGuardian,
    binary_to_program,
    compile_source,
    detect_language,
    render_source,
    verify_binary,
)
from dhriv.compiler import DhrivBinaryError
from dhriv.packager import build_package
from dhriv.triaxial_oop import TriAxialLargeScaleSearch
from dhriv.wave_electromagnetic import ElectromagneticBinaryModem, classify_em_band


def run(source: str, **kw) -> str:
    return DhrivInterpreter(**kw).run(source)


class LanguageCoreTests(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        self.assertEqual(run("print(1 + 2 * 3 ** 2)"), "19")
        self.assertEqual(run("print(7 / 2)"), "3.5")
        self.assertEqual(run("print(8 / 2)"), "4")

    def test_control_flow_and_functions(self):
        src = """
        fn fact(n) { if n <= 1 { return 1 } return n * fact(n - 1) }
        let total = 0
        for i in range(1, 6) { total = total + i }
        print(fact(5), total)
        """
        self.assertEqual(run(src), "120 15")

    def test_lists_maps_strings(self):
        src = 'let m = {"a": 1}; m.b = 2; let l = [1,2]; push(l, 3); print(len(l), m.a + m.b, upper("ok"))'
        self.assertEqual(run(src), "3 3 OK")

    def test_syntax_error_has_position(self):
        with self.assertRaises(DhrivSyntaxError) as ctx:
            run("let x = ")
        self.assertIn("L1", str(ctx.exception))

    def test_runtime_errors(self):
        with self.assertRaises(DhrivRuntimeError):
            run("print(1 / 0)")
        with self.assertRaises(DhrivRuntimeError):
            run("print(undefined_var)")

    def test_infinite_loop_is_bounded(self):
        with self.assertRaises(DhrivRuntimeError):
            DhrivInterpreter(max_steps=10_000).run("while true { }")


class TriAxialOOPTests(unittest.TestCase):
    SRC = """
    entity Base { let tag = "base"; fn who() { return "Base:" + self.tag } fn hello() { return "hello" } }
    entity Peer { fn peer_only() { return "peer" } }
    entity Bridge { fn bridged() { return "bridge" } }
    entity Child vertical Base horizontal Peer diagonal Bridge { fn hello() { return "child-hello" } }
    let c = new Child()
    """

    def test_vertical_inheritance_and_override(self):
        self.assertEqual(run(self.SRC + "print(c.who(), c.hello())"), "Base:base child-hello")

    def test_horizontal_and_diagonal_resolution(self):
        self.assertEqual(run(self.SRC + "print(c.peer_only(), c.bridged())"), "peer bridge")

    def test_missing_member_reports_axes(self):
        with self.assertRaises(DhrivRuntimeError) as ctx:
            run(self.SRC + "c.nothing()")
        self.assertIn("V/H/D", str(ctx.exception))

    def test_undeclared_parent_rejected(self):
        with self.assertRaises(DhrivRuntimeError):
            run("entity A vertical Missing { }")

    def test_dynamic_links_between_instances(self):
        src = """
        entity A { let x = 1 }
        entity B { fn ping() { return "pong from " + str(self.id) } let id = 7 }
        let a = new A(); let b = new B()
        a <-> b
        print(a.ping())
        let d = new A(); let t = new B()
        d </> t
        print(d.ping())
        """
        self.assertEqual(run(src), "pong from 7\npong from 7")

    def test_pipe_operator(self):
        src = "fn add(x, k) { return x + k } fn dbl(x) { return x * 2 } print(5 |> add(1) |> dbl)"
        self.assertEqual(run(src), "12")

    def test_instances_do_not_share_mutable_defaults(self):
        src = """
        entity Box { let items = [] }
        let a = new Box(); let b = new Box()
        push(a.items, 1)
        print(len(a.items), len(b.items))
        """
        self.assertEqual(run(src), "1 0")


class LargeScaleSearchTests(unittest.TestCase):
    def setUp(self):
        self.s = TriAxialLargeScaleSearch()
        self.s.register_node("n00", 0, 0, ["a"])
        self.s.register_node("n11", 1, 1, ["b"])
        self.s.register_node("n22", 2, 2, ["c"])
        self.s.register_node("n31", 3, 1, ["b"])

    def test_horizontal(self):
        self.assertEqual({r["node_id"] for r in self.s.search_horizontal(1)}, {"n11", "n31"})

    def test_vertical(self):
        self.assertEqual([r["node_id"] for r in self.s.search_vertical(2)], ["n22"])

    def test_diagonal_main_and_anti(self):
        ids = {r["node_id"] for r in self.s.search_diagonal(0, 0)}
        self.assertTrue({"n00", "n11", "n22"} <= ids)
        anti = {r["node_id"] for r in self.s.search_diagonal(2, 1)}  # h+v=3 -> n31? (3+1=4) no; h-v=1 -> n31
        self.assertIn("n31", anti)


class PolyglotTests(unittest.TestCase):
    IT = 'entita A { sia x = 1 }\nse vero { stampa("ciao") } // commento "se"\n'

    def test_detect(self):
        self.assertEqual(detect_language(self.IT), "it")
        self.assertEqual(detect_language("entity A { let x = 1 }"), "en")

    def test_italian_program_runs(self):
        self.assertEqual(run('se vero { stampa("ciao") } altrimenti { stampa("no") }'), "ciao")

    def test_render_preserves_strings_comments_and_semantics(self):
        out = render_source(self.IT, "en")
        self.assertIn("entity A { let x = 1 }", out)
        self.assertIn('// commento "se"', out)
        self.assertIn('"ciao"', out)
        self.assertEqual(detect_language(out), "en")
        for lang in ("en", "it", "es", "fr"):
            src = render_source("fn f(a) { if a > 0 { return true } else { return false } }\nprint(f(1))", lang)
            # `print` non è una parola chiave: il risultato deve restare identico
            self.assertEqual(run(src), "true")


class CompilerTests(unittest.TestCase):
    SRC = 'entity A { let v = 3 fn get() { return self.v } }\nlet a = new A()\nprint(a.get(), [1, {"k": 2.5}])'

    def test_roundtrip_same_output(self):
        blob = compile_source(self.SRC)
        program, meta = binary_to_program(blob)
        i = DhrivInterpreter()
        self.assertEqual(i.run_program(program), run(self.SRC))
        self.assertEqual(meta["language"], "en")

    def test_tamper_detected(self):
        blob = bytearray(compile_source(self.SRC))
        blob[-3] ^= 0xFF
        self.assertFalse(verify_binary(bytes(blob)))
        with self.assertRaises(DhrivBinaryError):
            binary_to_program(bytes(blob))
        self.assertFalse(verify_binary(b"not a dhrb"))

    def test_incremental_autocompile(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "p.dhr"
            src.write_text("print(1)", encoding="utf-8")
            ac = AutoCompiler(Path(d) / "cache")
            self.assertTrue(ac.compile_file(src).rebuilt)
            self.assertFalse(ac.compile_file(src).rebuilt)
            src.write_text("print(2)", encoding="utf-8")
            self.assertTrue(ac.compile_file(src).rebuilt)


class WaveTests(unittest.TestCase):
    def test_band_and_wavelength(self):
        self.assertIn("UHF", classify_em_band(2.4e9))
        self.assertEqual(classify_em_band(1000, "ACOUSTIC"), "AUDIBLE_ACOUSTIC")
        self.assertAlmostEqual(float(run("print(wavelength(3e8))")), 299792458.0 / 3e8, places=4)

    def test_modem_roundtrip_with_noise(self):
        m = ElectromagneticBinaryModem()
        payload = bytes(range(40))
        for scheme in ("BFSK", "BPSK"):
            noisy = m.inject_channel_noise(m.modulate_binary(payload, scheme), snr_db=10)
            out, info = m.demodulate_binary(noisy, scheme)
            self.assertEqual(out, payload)
            self.assertTrue(info["crc32_verified"])

    def test_modem_wav_roundtrip(self):
        m = ElectromagneticBinaryModem()
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "t.wav"
            m.export_wav(m.modulate_binary(b"dhriv-over-the-air"), wav)
            out, _ = m.demodulate_binary(m.import_wav(wav))
            self.assertEqual(out, b"dhriv-over-the-air")

    def test_fec_corrects_flipped_bits(self):
        from dhriv.wave_electromagnetic import _Hamming74FEC as F

        bits = F.encode_bytes(b"Z")
        bits[3] ^= 1
        bits[10] ^= 1
        data, fixed = F.decode_bits(bits)
        self.assertEqual((data, fixed), (b"Z", 2))

    def test_dhrb_binary_travels_over_wave_and_runs(self):
        m = ElectromagneticBinaryModem()
        blob = compile_source('print("eseguito dopo trasmissione")')
        out, _ = m.demodulate_binary(m.inject_channel_noise(m.modulate_binary(blob), snr_db=12))
        program, _ = binary_to_program(out)
        self.assertEqual(DhrivInterpreter().run_program(program), "eseguito dopo trasmissione")


class SelfLearningTests(unittest.TestCase):
    FIB = "fn fib(n) { if n < 2 { return n } return fib(n - 1) + fib(n - 2) }\nprint(fib(%d))"

    def test_learns_and_memoizes_pure_function(self):
        learner = SelfLearningEngine()
        t0 = time.perf_counter()
        out = run(self.FIB % 27, learner=learner)
        self.assertEqual(out, "196418")
        self.assertLess(time.perf_counter() - t0, 5.0)  # senza memo: ~800k chiamate
        self.assertIn("fib", learner.report()["memoized_functions"])

    def test_impure_function_is_never_memoized(self):
        src = """
        let counter = 0
        fn bump(x) { counter = counter + 1 return counter }
        let i = 0
        while i < 20 { bump(1) i = i + 1 }
        print(bump(1))
        """
        learner = SelfLearningEngine()
        self.assertEqual(run(src, learner=learner), "21")
        self.assertNotIn("bump", learner.report()["memoized_functions"])

    def test_knowledge_persists_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "k.json"
            l1 = SelfLearningEngine(path)
            run(self.FIB % 15, learner=l1)
            l1.finish_run()
            l2 = SelfLearningEngine(path)
            self.assertIn("fib", l2.knowledge["hot_pure"])
            self.assertEqual(l2.knowledge["runs"], 1)


class GuardianWatchdogTests(unittest.TestCase):
    def test_guardian_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (pkg / "a.py").write_text("x = 1\n", encoding="utf-8")
            g = WatchGuardian(pkg, Path(d) / "state")
            self.assertFalse(g.verify().sealed)
            g.seal()
            self.assertTrue(g.verify().ok)
            (pkg / "a.py").write_text("x = 2\n", encoding="utf-8")
            (pkg / "b.py").write_text("y = 1\n", encoding="utf-8")
            r = g.verify()
            self.assertFalse(r.ok)
            self.assertEqual((r.modified, r.added), (["a.py"], ["b.py"]))

    def test_guardian_scans_sensitive_calls(self):
        from dhriv import DhrivInterpreter as _  # noqa: F401
        from dhriv.parser import DhrivParser

        warns = WatchGuardian.scan_program(DhrivParser.parse_source('legacy_run("ls")'))
        self.assertEqual(len(warns), 1)

    def test_legacy_run_requires_opt_in(self):
        with self.assertRaises(DhrivRuntimeError):
            run('legacy_run("echo hi")')
        out = run('print(legacy_run("echo hi").stdout)', allow_legacy=True)
        self.assertEqual(out.strip(), "hi")

    def test_watchdog_ok_and_deterministic_error(self):
        ok = WatchDog().supervise("print(1 + 1)")
        self.assertTrue(ok.ok)
        self.assertEqual((ok.output, ok.attempts), ("2", 1))
        bad = WatchDog().supervise("print(1 / 0)")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.attempts, 1)  # errore deterministico: nessun riavvio inutile

    def test_watchdog_stops_runaway_program(self):
        dog = WatchDog(timeout_s=0.5, max_restarts=1)
        # budget enorme: solo il WatchDog può fermarlo
        res = dog.supervise("let i = 0 while true { i = i + 1 }")
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertEqual(res.attempts, 2)

    def test_watchdog_recompiles_on_change(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "w.dhr"
            src.write_text("print(1)", encoding="utf-8")
            events = list(WatchDog().watch_events(src, AutoCompiler(Path(d) / "c"), 0.0, 1))
            self.assertEqual(events[0]["event"], "recompiled")


class PackagerTests(unittest.TestCase):
    def test_universal_package(self):
        with tempfile.TemporaryDirectory() as d:
            dhrb = Path(d) / "p.dhrb"
            dhrb.write_bytes(compile_source('print("x")'))
            z = build_package(dhrb, "Demo", "universal", Path(d) / "out")
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                self.assertIn("macos/Demo.app/Contents/MacOS/launcher", names)
                self.assertIn("windows/run.bat", names)
                self.assertIn("linux/run.sh", names)
                self.assertIn("PACKAGE_MANIFEST.json", names)
                mode = zf.getinfo("linux/run.sh").external_attr >> 16
                self.assertTrue(mode & 0o111)

    def test_tampered_binary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.dhrb"
            bad.write_bytes(b"DHRB" + b"\x00" * 60)
            with self.assertRaises(ValueError):
                build_package(bad, "X", "linux", Path(d))


if __name__ == "__main__":
    unittest.main()
