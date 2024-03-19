use std::time::Instant;

use tfhe::{
    integer::{
        gen_keys_radix, wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn main() {
    let nb_block = 4;
    let msg = 14;
    let (cks, sks) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_block);
    let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks, &WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS);

    let ct = cks.encrypt(msg);
    let ct = wopbs_key.keyswitch_to_wopbs_params(&sks, &ct);

    let lut_lsb = wopbs_key.generate_lut_radix(&ct, |x| u64::pow(x, 2) % (1 << 4));
    let lut_msb = wopbs_key.generate_lut_radix(&ct, |x| (u64::pow(x, 2) >> 4) % (1 << 4));

    let start = Instant::now();
    let (ct_res_lsb, ct_res_msb) = rayon::join(
        || {
            let ct_res_lsb = wopbs_key.wopbs(&ct, &lut_lsb);
            wopbs_key.keyswitch_to_pbs_params(&ct_res_lsb)
        },
        || {
            let ct_res_msb = wopbs_key.wopbs(&ct, &lut_msb);
            wopbs_key.keyswitch_to_pbs_params(&ct_res_msb)
        },
    );

    let mut lsb_blocks = ct_res_lsb.clone().into_blocks();
    let msb_blocks = ct_res_msb.clone().into_blocks();
    lsb_blocks.extend(msb_blocks);
    let _ct_res = RadixCiphertext::from_blocks(lsb_blocks);
    let duration = start.elapsed();
    println!("PBS time: {:?}", duration);
    let res_lsb: u64 = cks.decrypt(&ct_res_lsb);
    let res_msb: u64 = cks.decrypt(&ct_res_msb);

    assert_eq!(res_lsb, u64::pow(msg, 2) % (1 << 4));
    assert_eq!(res_msb, (u64::pow(msg, 2) >> 4) % (1 << 4));
    assert_eq!((res_msb << 4) + res_lsb, u64::pow(msg, 2));
}
