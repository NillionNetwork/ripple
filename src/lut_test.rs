use std::time::Instant;

use tfhe::{
    integer::{gen_keys_radix, wopbs::*},
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn main() {
    let nb_block = 8;
    let (cks, sks) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_block);
    let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks, &WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS);
    let ct = cks.encrypt(2_u64);
    let ct = wopbs_key.keyswitch_to_wopbs_params(&sks, &ct);
    let lut = wopbs_key.generate_lut_radix(&ct, |x| 5 + 2 * x);
    let start = Instant::now();
    let ct_res = wopbs_key.wopbs(&ct, &lut);
    let ct_res = wopbs_key.keyswitch_to_pbs_params(&ct_res);
    let duration = start.elapsed();
    println!("PBS time: {:?}", duration);
    let res: u64 = cks.decrypt(&ct_res);
    assert_eq!(res, 9);
}
