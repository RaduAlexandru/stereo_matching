/*
 * Author: Konstantin Schauwecker
 * Year:   2012
 */

#include "stereo_test/simd.h"

#define S16(x) SIMD::scalar16NonLookup(x)

namespace sparsestereo {
	v16qi SIMD::v16Constants[256] __attribute__((aligned (16))) = {
		S16(0), S16(1), S16(2), S16(3), S16(4), S16(5), S16(6), S16(7), S16(8), S16(9), S16(10), S16(11), S16(12), S16(13), S16(14), S16(15),
		S16(16), S16(17), S16(18), S16(19), S16(20), S16(21), S16(22), S16(23), S16(24), S16(25), S16(26), S16(27), S16(28), S16(29), S16(30), S16(31),
		S16(32), S16(33), S16(34), S16(35), S16(36), S16(37), S16(38), S16(39), S16(40), S16(41), S16(42), S16(43), S16(44), S16(45), S16(46), S16(47),
		S16(48), S16(49), S16(50), S16(51), S16(52), S16(53), S16(54), S16(55), S16(56), S16(57), S16(58), S16(59), S16(60), S16(61), S16(62), S16(63),
		S16(64), S16(65), S16(66), S16(67), S16(68), S16(69), S16(70), S16(71), S16(72), S16(73), S16(74), S16(75), S16(76), S16(77), S16(78), S16(79),
		S16(80), S16(81), S16(82), S16(83), S16(84), S16(85), S16(86), S16(87), S16(88), S16(89), S16(90), S16(91), S16(92), S16(93), S16(94), S16(95),
		S16(96), S16(97), S16(98), S16(99), S16(100), S16(101), S16(102), S16(103), S16(104), S16(105), S16(106), S16(107), S16(108), S16(109), S16(110), S16(111),
		S16(112), S16(113), S16(114), S16(115), S16(116), S16(117), S16(118), S16(119), S16(120), S16(121), S16(122), S16(123), S16(124), S16(125), S16(126), S16(127),
		S16(128), S16(129), S16(130), S16(131), S16(132), S16(133), S16(134), S16(135), S16(136), S16(137), S16(138), S16(139), S16(140), S16(141), S16(142), S16(143),
		S16(144), S16(145), S16(146), S16(147), S16(148), S16(149), S16(150), S16(151), S16(152), S16(153), S16(154), S16(155), S16(156), S16(157), S16(158), S16(159),
		S16(160), S16(161), S16(162), S16(163), S16(164), S16(165), S16(166), S16(167), S16(168), S16(169), S16(170), S16(171), S16(172), S16(173), S16(174), S16(175),
		S16(176), S16(177), S16(178), S16(179), S16(180), S16(181), S16(182), S16(183), S16(184), S16(185), S16(186), S16(187), S16(188), S16(189), S16(190), S16(191),
		S16(192), S16(193), S16(194), S16(195), S16(196), S16(197), S16(198), S16(199), S16(200), S16(201), S16(202), S16(203), S16(204), S16(205), S16(206), S16(207),
		S16(208), S16(209), S16(210), S16(211), S16(212), S16(213), S16(214), S16(215), S16(216), S16(217), S16(218), S16(219), S16(220), S16(221), S16(222), S16(223),
		S16(224), S16(225), S16(226), S16(227), S16(228), S16(229), S16(230), S16(231), S16(232), S16(233), S16(234), S16(235), S16(236), S16(237), S16(238), S16(239),
		S16(240), S16(241), S16(242), S16(243), S16(244), S16(245), S16(246), S16(247), S16(248), S16(249), S16(250), S16(251), S16(252), S16(253), S16(254), S16(255)
	};
}
