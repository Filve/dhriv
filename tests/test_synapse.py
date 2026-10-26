import unittest

from dhriv import (
    AxisCode,
    BinaryALU,
    BytecodeCompiler,
    BytecodeModule,
    DhrivInterpreter,
    DhrivSyntaxError,
    DhrivVM,
    InstructionWord,
    OpCode,
    SelfAttentionLayer,
    SelfLearningEngine,
    Tensor,
    TriAxialNetwork,
    binary_to_bytecode,
    compile_source,
    disassemble_binary,
    optimize_ast,
    pack_weights_binary,
    unpack_weights_binary,
)
from dhriv.ai_engine import cross_entropy_loss, mse_loss
from dhriv.ast_nodes import ExprStmtNode, LiteralNode
from dhriv.binary_core import eval_gate, full_adder, half_adder, mux2
from dhriv.parser import DhrivParser


def run_both(test_case: unittest.TestCase, source: str, **kw) -> str:
    out_ast = DhrivInterpreter(**kw).run(source)
    out_vm = DhrivVM(**kw).run_source(source)
    test_case.assertEqual(out_ast, out_vm, "Discrepanza tra motore AST e DhrivVM a 64 bit!")
    return out_vm


class BinaryCoreTests(unittest.TestCase):
    def test_universal_nand_derived_gates(self):
        self.assertEqual([eval_gate("NAND", a, b) for a in (0, 1) for b in (0, 1)], [1, 1, 1, 0])
        self.assertEqual([eval_gate("AND", a, b) for a in (0, 1) for b in (0, 1)], [0, 0, 0, 1])
        self.assertEqual([eval_gate("OR", a, b) for a in (0, 1) for b in (0, 1)], [0, 1, 1, 1])
        self.assertEqual([eval_gate("NOR", a, b) for a in (0, 1) for b in (0, 1)], [1, 0, 0, 0])
        self.assertEqual([eval_gate("XOR", a, b) for a in (0, 1) for b in (0, 1)], [0, 1, 1, 0])
        self.assertEqual([eval_gate("XNOR", a, b) for a in (0, 1) for b in (0, 1)], [1, 0, 0, 1])
        self.assertEqual((eval_gate("NOT", 0), eval_gate("NOT", 1)), (1, 0))
        self.assertEqual((mux2(0, 1, 0), mux2(1, 1, 0)), (1, 0))

    def test_adders_and_alu_flags(self):
        self.assertEqual(half_adder(1, 1), (0, 1))
        self.assertEqual(full_adder(1, 1, 1), (1, 1))

        alu = BinaryALU(width=8)
        r_add = alu.execute("ADD", 42, 16)
        self.assertEqual((r_add["value"], r_add["bitstring"]), (58, "00111010"))
        self.assertEqual(r_add["flags"], {"Z": 0, "C": 0, "N": 0, "V": 0})

        # Signed overflow: 127 + 1 = -128 in complemento a 2 su 8 bit -> V=1, N=1
        r_ovf = alu.execute("ADD", 127, 1)
        self.assertEqual(r_ovf["signed"], -128)
        self.assertEqual((r_ovf["flags"]["V"], r_ovf["flags"]["N"]), (1, 1))

        # Carry & Zero: 255 + 1 = 0 -> Z=1, C=1
        r_cz = alu.execute("ADD", 255, 1)
        self.assertEqual((r_cz["value"], r_cz["flags"]["Z"], r_cz["flags"]["C"]), (0, 1, 1))

        # Sottrazione in complemento a 2: 5 - 9 = -4 (0xFC)
        r_sub = alu.execute("SUB", 5, 9)
        self.assertEqual((r_sub["signed"], r_sub["hex"]), (-4, "0xFC"))

        # Moltiplicatore binario shift-and-add: 13 * 11 = 143
        self.assertEqual(alu.execute("MUL", 13, 11)["value"], 143)

    def test_binary_hex_octal_literals_and_bitwise_in_dhriv(self):
        src = """
        let a = 0b1010_0110
        let b = 0x0F
        let c = 0o17
        print(a, b == c, (a & b), (0b0101 | 0b1010), (0b1100 ^ 0b1010), (1 << 5), (64 >> 2), ~0)
        """
        self.assertEqual(run_both(self, src), "166 true 6 15 6 32 16 -1")

        with self.assertRaises(DhrivSyntaxError):
            DhrivParser.parse_source("let bad = 0b2")

    def test_instruction_word_64bit_roundtrip(self):
        w = InstructionWord(OpCode.TRIAXIAL_OP, AxisCode.DIAGONAL, flags=3, arg_a=512, arg_b=-42)
        packed = w.pack()
        self.assertEqual(len(packed), 8)
        restored = InstructionWord.unpack(packed)
        self.assertEqual(restored, w)
        self.assertEqual(len(w.format_bits().replace("_", "")), 64)


class BytecodeVMTests(unittest.TestCase):
    def test_constant_folding(self):
        prog = optimize_ast(DhrivParser.parse_source("1 + 2 * 3 ** 2"))
        stmt = prog.statements[0]
        self.assertIsInstance(stmt, ExprStmtNode)
        self.assertIsInstance(stmt.expr, LiteralNode)
        self.assertEqual(stmt.expr.value, 19)

        # Divisione per zero NON viene piegata staticamente per preservare l'errore a runtime
        prog_div0 = optimize_ast(DhrivParser.parse_source("1 / 0"))
        self.assertNotIsInstance(prog_div0.statements[0].expr, LiteralNode)

    def test_bytecode_binary_serialization_and_disasm(self):
        src = """
        entity Node { fn f(x) => x + 1 }
        let n = new Node()
        print(41 |> n.f)
        """
        mod = BytecodeCompiler.compile_source(src)
        raw = mod.to_bytes()
        mod2 = BytecodeModule.from_bytes(raw)
        self.assertEqual(DhrivVM().run_module(mod2), "42")

        dhrb = compile_source(src)
        bc_from_dhrb, meta = binary_to_bytecode(dhrb)
        self.assertEqual(DhrivVM().run_module(bc_from_dhrb), "42")
        dis = disassemble_binary(dhrb, show_bits=True)
        self.assertIn("TRIAXIAL_OP.VERTICAL", dis)
        self.assertIn("DECL_ENTITY", dis)

    def test_vm_self_learning_integration(self):
        src = "fn fib(n) { if n < 2 { return n } return fib(n - 1) + fib(n - 2) }\nprint(fib(24))"
        learner = SelfLearningEngine()
        vm = DhrivVM(learner=learner)
        self.assertEqual(vm.run_source(src), "46368")
        self.assertIn("fib", learner.report()["memoized_functions"])


class EnhancedTriAxialOOPTests(unittest.TestCase):
    def test_callable_entities_and_lambdas_in_vertical_pipes(self):
        src = """
        entity Doubler {
            let factor = 2
            fn init(f) { self.factor = f }
            fn forward(x) => x * self.factor
        }
        let d3 = new Doubler(3)
        let nums = [1, 2, 3, 4]
        let mapped = map_list(nums, fn(v) => v |> d3)
        let total = reduce_list(mapped, fn(acc, v) => acc + v, 0)
        print(mapped, total)
        """
        self.assertEqual(run_both(self, src), "[3, 6, 9, 12] 30")

    def test_explicit_axis_call_and_swarm_broadcast(self):
        src = """
        entity VParent { fn role() => "V" }
        entity HPeer { fn role() => "H" }
        entity DBridge { fn role() => "D" }
        entity Hybrid vertical VParent horizontal HPeer diagonal DBridge {
            let tag = ""
            fn init(t) { self.tag = t }
            fn role() => "SELF:" + self.tag
            fn ping(k) => self.tag + "=" + str(k)
        }
        let a = new Hybrid("A")
        let b = new Hybrid("B")
        let c = new Hybrid("C")
        a <-> b
        a </> c
        print(a.role(), axis_call(a, "VERTICAL", "role"), axis_call(a, "HORIZONTAL", "role"), axis_call(a, "DIAGONAL", "role"))
        print(axis_broadcast(a, "HORIZONTAL", "ping", [7]))
        print(axis_broadcast(a, "ALL", "ping", [9]))
        let info = coords_of(a)
        print(info.v, info.h_links, info.d_links)
        """
        expected = 'SELF:A V H D\n["B=7"]\n["B=9", "C=9"]\n1 1 1'
        self.assertEqual(run_both(self, src), expected)


class AIEngineTests(unittest.TestCase):
    def test_autograd_finite_difference_verification(self):
        x = Tensor([[0.4, -0.7, 1.1], [-0.3, 0.8, 0.2]], requires_grad=True)
        w = Tensor([[0.5, -0.2], [0.3, 0.9], [-0.6, 0.4]], requires_grad=True)
        b = Tensor([[0.1, -0.15]], requires_grad=True)
        target = Tensor([[0.7, 0.1], [-0.2, 0.8]])

        def compute_loss() -> Tensor:
            h = x.matmul(w).add(b).gelu()
            s = h.softmax()
            return mse_loss(s, target)

        loss = compute_loss()
        loss.backward()
        analytic_w_grad = [list(row) for row in w.grad]
        analytic_b_grad = [list(row) for row in b.grad]

        eps = 1e-5
        for r in range(w.rows):
            for c in range(w.cols):
                orig = w.data[r][c]
                w.data[r][c] = orig + eps
                lp = compute_loss().item()
                w.data[r][c] = orig - eps
                lm = compute_loss().item()
                w.data[r][c] = orig
                numeric = (lp - lm) / (2.0 * eps)
                self.assertAlmostEqual(analytic_w_grad[r][c], numeric, places=4)

        for c in range(b.cols):
            orig = b.data[0][c]
            b.data[0][c] = orig + eps
            lp = compute_loss().item()
            b.data[0][c] = orig - eps
            lm = compute_loss().item()
            b.data[0][c] = orig
            numeric = (lp - lm) / (2.0 * eps)
            self.assertAlmostEqual(analytic_b_grad[0][c], numeric, places=4)

    def test_triaxial_tensor_operators_in_language(self):
        src = """
        let A = tensor([[1, 2], [3, 4]])
        let B = tensor([[2, 0], [1, 2]])
        // </> su Tensor esegue proiezione diagonale (matmul)
        let P = A </> B
        // <-> su Tensor esegue interferenza simmetrica (A + B) / 2
        let M = A <-> B
        print(P.values, M.values)
        """
        self.assertEqual(run_both(self, src), "[[4, 4], [10, 8]] [[1.5, 1], [2, 3]]")

    def test_triaxial_network_learns_xor(self):
        src = """
        let X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        let Y = tensor([[0], [1], [1], [0]])
        let net = nn_triaxial(2, 8, 1, 2, "sigmoid", 7)
        let opt = nn_adam(net, 0.05)
        for ep in range(0, 200) {
            opt_zero_grad(opt)
            let out = X |> net
            let loss = bce_loss(out, Y)
            backward(loss)
            opt_step(opt)
        }
        let p = X |> net
        print(round(p[0][0]), round(p[1][0]), round(p[2][0]), round(p[3][0]))
        """
        self.assertEqual(run_both(self, src), "0 1 1 0")

    def test_self_attention_and_cross_entropy(self):
        attn = SelfAttentionLayer(d_model=4, seed=11)
        x = Tensor([[1.0, 0.0, 0.5, -0.5], [0.2, 0.8, -0.1, 0.4]], requires_grad=True)
        out = attn(x)
        self.assertEqual(out.shape, [2, 4])
        loss = cross_entropy_loss(out, [0, 3])
        loss.backward()
        # Verifica che i gradienti fluiscano fino a Q, K, V e all'input
        self.assertTrue(any(abs(v) > 0.0 for row in attn.W_q.grad for v in row))
        self.assertTrue(any(abs(v) > 0.0 for row in x.grad for v in row))

    def test_neuro_symbolic_memory_and_em_weight_transmission(self):
        src = """
        let mem = ai_memory()
        mem_store(mem, "alpha", [1.0, 0.0, 0.0], "first")
        mem_store(mem, "beta", [0.0, 1.0, 0.0], "second")
        let top = mem_recall(mem, [0.9, 0.1, 0.0], 1)[0]

        let layer = nn_linear(3, 2, 99)
        let tx = em_transmit_ai(layer, 15, "BFSK")
        print(top.key, top.payload, tx.ok, len(tx.tensors))
        """
        self.assertEqual(run_both(self, src), "alpha first true 2")

    def test_weight_binary_codec_tamper_detection(self):
        net = TriAxialNetwork(2, 4, 1, depth=1, seed=3)
        blob = bytearray(pack_weights_binary(net.parameters()))
        restored = unpack_weights_binary(bytes(blob))
        self.assertEqual(len(restored), len(net.parameters()))
        blob[10] ^= 0xFF
        with self.assertRaises(ValueError):
            unpack_weights_binary(bytes(blob))


if __name__ == "__main__":
    unittest.main()
